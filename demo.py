import argparse, cv2, os, requests, json, time, io
from paddleocr import PaddleOCR
from cv2 import dnn_superres
import numpy as np
import onnxruntime as ort
import torch, pickle, math
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Colors
from spandrel import ModelLoader, ImageModelDescriptor
from pyzbar.pyzbar import decode
from PIL import Image

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")
cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(
    "output.avi",
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    30,
    (frame_width, frame_height),
)
scaled_size = (frame_width, frame_height)

# Load COCO class names
classes = ["Front", "Back", "Side"]

# Create color palette
color_palette = Colors()


def build_model(onnx_model):
    """Builds the ONNX model session."""
    session = ort.InferenceSession(
        onnx_model,
        providers=(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        ),
    )
    ndtype = np.half if session.get_inputs()[0].type == "tensor(float16)" else np.single
    model_height, model_width = [x.shape for x in session.get_inputs()][0][-2:]
    return session, ndtype, model_height, model_width


def preprocess(img, model_height, model_width, ndtype):
    """Pre-processes the input image."""
    shape = img.shape[:2]  # original image shape
    new_shape = (model_height, model_width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=ndtype) / 255.0
    img_process = img[None] if len(img.shape) == 3 else img
    return img_process, ratio, (pad_w, pad_h)


def postprocess(preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
    """Post-process the prediction."""
    x, protos = preds[0], preds[1]  # Two outputs: predictions and protos
    x = np.einsum("bcn->bnc", x)
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
    x = np.c_[
        x[..., :4],
        np.amax(x[..., 4:-nm], axis=-1),
        np.argmax(x[..., 4:-nm], axis=-1),
        x[..., -nm:],
    ]
    x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

    if len(x) > 0:
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]
        x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        x[..., :4] /= min(ratio)
        x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
        x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])
        masks = process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
        segments = masks2segments(masks)
        return x[..., :6], segments, masks  # boxes, segments, masks
    else:
        return [], [], []


def masks2segments(masks):
    """Takes a list of masks(n,h,w) and returns a list of segments(n,xy)."""
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
            0
        ]  # CHAIN_APPROX_SIMPLE
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def crop_mask(masks, boxes):
    """Crops a mask to the bounding box."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, im0_shape):
    """Applies the mask to the bounding boxes."""
    c, mh, mw = protos.shape
    masks = (
        np.matmul(masks_in, protos.reshape((c, -1)))
        .reshape((-1, mh, mw))
        .transpose(1, 2, 0)
    )  # HWN
    masks = np.ascontiguousarray(masks)
    masks = scale_mask(masks, im0_shape)  # re-scale mask
    masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)


def scale_mask(masks, im0_shape, ratio_pad=None):
    """Resizes the mask to the original image size."""
    im1_shape = masks.shape[:2]
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(
            im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
        )  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
            im1_shape[0] - im0_shape[0] * gain
        ) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(
        round(im1_shape[1] - pad[0] + 0.1)
    )
    if len(masks.shape) < 2:
        raise ValueError(
            f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
        )
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(
        masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
    )  # INTER_CUBIC would be better
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


def draw_and_visualize(im, boxes, segments, album_name):
    """Draw and visualize results."""
    im_canvas = im.copy()
    for (*box, conf, cls_), segment in zip(boxes, segments):
        # Check if the segment is valid
        if segment.size > 0:
            segment = segment.astype(np.int32)  # Ensure segment points are integers
            # Draw contour and fill mask
            cv2.polylines(
                im, np.int32([segment]), True, (255, 255, 255), 2
            )  # white borderline
            cv2.fillPoly(
                im_canvas, np.int32([segment]), color_palette(int(cls_), bgr=True)
            )

            # Draw bbox rectangle
            cv2.rectangle(
                im,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color_palette(int(cls_), bgr=True),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                im,
                f"{album_name}: {conf:.3f}",
                (int(box[0]), int(box[1] - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color_palette(int(cls_), bgr=True),
                2,
                cv2.LINE_AA,
            )

    # Mix image
    im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
    cv2.namedWindow("Name-Searcher", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Name-Searcher", scaled_size)
    cv2.imshow("Name-Searcher", im)


def get_full_album_name(jsonstr):
    """Reads MusicBrainz JSON for album details."""
    j = json.loads(jsonstr)
    outstr = (
        j["releases"][0]["title"] + " - " + j["releases"][0]["artist-credit"][0]["name"]
    )
    try:
        outstr += (
            j["releases"][0]["artist-credit"][0]["joinphrase"]
            + " "
            + j["releases"][0]["artist-credit"][1]["name"]
        )
    except KeyError as e:
        pass
    except IndexError as e:
        pass
    try:
        outstr += (
            j["releases"][0]["artist-credit"][1]["joinphrase"]
            + " "
            + j["releases"][0]["artist-credit"][2]["name"]
        )
    except KeyError as e:
        pass
    except IndexError as e:
        pass
    return outstr


lastcat = ""
lastjson = ""


def mb_request(catnum, request_type):
    """Requests MusicBrainz for album with catalog number."""
    global lastcat
    global lastjson
    if lastcat != catnum:
        lastcat = catnum
        lastjson = requests.get(
            'https://musicbrainz.org/ws/2/release?query={}:"{}"&limit=1&fmt=json'.format(
                request_type, catnum
            )
        ).text
    return lastjson


def upscale_image(image, model_path="weights/8x_NMKD-Typescale_175k.pth"):
    # """Upscales an image"""
    # # load a model from disk
    # model = ModelLoader().load_from_file(model_path)
    # assert isinstance(
    #     model, ImageModelDescriptor
    # ), "Loaded model is not an image-to-image model."

    # # Prepare the model for inference
    # model = model.cuda().eval()  # Move model to GPU (if available) and set to eval mode

    # # Convert OpenCV image (BGR) to PyTorch tensor (CHW, normalized)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_tensor = (
    #     torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # )
    # image_tensor = image_tensor.cuda()  # Move to GPU

    # # Process the image with the model
    # with torch.no_grad():
    #     output_tensor = model(image_tensor)

    # # Convert output tensor back to OpenCV image format (HWC, 0-255 range, BGR)
    # output_image = (
    #     output_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
    # ).astype("uint8")
    # output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # cv2.imwrite(f"cropped_images/upscaled.jpg", output_image)

    # return output_image_bgr
    return image


def decode_barcode(image):
    """Detect and decode barcodes from the image."""

    barcode_images = detect_barcode(image)
    if not barcode_images:
        return None
    upscaled_barcode = upscale_image(barcode_images[0])
    barcodes = decode(upscaled_barcode)
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")  # Convert to string
        barcode_type = barcode.type  # E.g., CODE128, EAN13, etc.
        print(f"Detected barcode: {barcode_data} (Type: {barcode_type})")
        print(barcode.rect)
        return barcode_data  # Return the first barcode found
    return None  # No barcode detected


def detect_barcode(
    image, model_path="weights/yolo8n-barcode.onnx", confidence_thres=0.5, iou_thres=0.5
):
    # Load the ONNX model
    session, ndtype, input_height, input_width = build_model(model_path)

    # Preprocess the input image
    img_height, img_width = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_width, input_height))
    img_normalized = img_resized / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_data = np.expand_dims(img_transposed, axis=0).astype(np.float32)

    # Perform inference
    outputs = session.run(None, {session.get_inputs()[0].name: img_data})
    detections = np.transpose(np.squeeze(outputs[0]))

    # Post-process the detections
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    boxes, scores, class_ids = [], [], []

    for detection in detections:
        class_scores = detection[4:]
        max_score = np.amax(class_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(class_scores)
            x, y, w, h = detection[:4]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            boxes.append([left, top, width, height])
            scores.append(max_score)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    cropped_images = []

    if indices is not None:
        for i in indices.flatten():
            left, top, width, height = boxes[i]
            cropped_image = image[
                max(0, top) : min(img_height, top + height),
                max(0, left) : min(img_width, left + width),
            ]
            cropped_images.append(cropped_image)

        for idx, cropped_image in enumerate(cropped_images):
            cv2.imwrite(f"cropped_images/object_{idx}.jpg", cropped_image)

        return cropped_images
    return None


def extract_text_with_paddleocr(image):
    """Takes an OpenCV image, runs PaddleOCR, and returns the extracted text strings."""

    # Convert OpenCV image to RGB format (PaddleOCR requires RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run OCR on the image
    results = ocr.ocr(image_rgb, cls=True)

    # Extract text strings from results
    text_strings = [line[1][0] for line in results[0]]

    return text_strings


def scan_cropped_masked_image(
    im,
    segments,
    boxes,
    back_threshold=0.9,
    side_threshold=0.7,
    output_dir="cropped_images",
):
    """Crops the image to the mask and scans it if confidence is greater than the threshold."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (mask, box) in enumerate(zip(segments, boxes)):
        conf = box[4]  # Confidence is the 5th element in the box array
        object_name = classes[int(box[5])]
        # Create a binary mask for the current object
        binary_mask = mask.astype(np.uint8) * 255  # Convert mask to binary (0 or 255)

        # Ensure the mask is the same size as the original image
        mask_resized = cv2.resize(
            binary_mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR
        )

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(im, im, mask=mask_resized)

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_image = masked_image[y1:y2, x1:x2]

        # Save the cropped image
        output_path = os.path.join(output_dir, f"cropped_mask_{object_name}.png")
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved cropped image: {output_path}")
        if conf > side_threshold and object_name == "Side":
            try:
                print("1")
                ocr_list = extract_text_with_paddleocr(cropped_image)
            except TypeError as e:
                print("2")
                upscaled_image = upscale_image(cropped_image)
                try:
                    print("3")
                    ocr_list = extract_text_with_paddleocr(upscaled_image)
                except TypeError as e:
                    print("4")
                    return "Side"
            print("5")
            sorted_list = sorted(
                ocr_list,
                key=lambda s: sum(c.isdigit() for c in s) / len(s) if len(s) > 0 else 0,
            )
            print("6")
            catnum = sorted_list[-1]
            print("7")
            if any(char.isdigit() for char in catnum):
                print(f"Catalog Number: {catnum}")
                jsonstr = mb_request(catnum, "catno")
                try:
                    print("8")
                    return get_full_album_name(jsonstr)
                except IndexError as e:
                    return catnum
            else:
                print("Catalog Number not found")
                return catnum

        if conf > back_threshold and object_name == "Back":
            # Add barcode scanning here
            barcodenum = decode_barcode(cropped_image)
            if barcodenum:
                print(f"Barcode: {barcodenum}")
                jsonstr = mb_request(barcodenum, "barcode")
                try:
                    return get_full_album_name(jsonstr)
                except IndexError as e:
                    return barcodenum
            else:
                print("Barcode not found")
            return "Back"

        if conf > back_threshold and object_name == "Front":
            return "Front"
        return ""


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--img", type=str, help="Path to input image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    # Build model
    session, ndtype, model_height, model_width = build_model(args.model)

    print(args.img)
    while True:
        if args.img is None:
            # Read image by OpenCV
            success, img = cap.read()
            if not success:
                break
        else:
            img = cv2.imread(args.img)
            height, width = img.shape[:2]
            if height < width:
                scale_factor = 720 / height
            else:
                scale_factor = 720 / width

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            scaled_size = (new_width, new_height)

            img = cv2.resize(img, scaled_size, interpolation=cv2.INTER_AREA)

        # Preprocess the image
        img_processed, ratio, (pad_w, pad_h) = preprocess(
            img, model_height, model_width, ndtype
        )

        # Inference
        preds = session.run(None, {session.get_inputs()[0].name: img_processed})
        boxes, segments, _ = postprocess(
            preds,
            img,
            ratio,
            pad_w,
            pad_h,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )

        # Save cropped masked images if confidence is greater than 90%
        name = scan_cropped_masked_image(img, segments, boxes)

        # Draw and visualize results
        draw_and_visualize(img, boxes, segments, name)

        # Break the loop on key press
        if cv2.waitKey(1) & 0xFF == ord("1"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
