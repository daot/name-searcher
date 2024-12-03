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
ocr = PaddleOCR(use_angle_cls=True, lang='en') 
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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=ndtype) / 255.0
    img_process = img[None] if len(img.shape) == 3 else img
    return img_process, ratio, (pad_w, pad_h)

def postprocess(preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
    """Post-process the prediction."""
    x, protos = preds[0], preds[1]  # Two outputs: predictions and protos
    x = np.einsum("bcn->bnc", x)
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
    x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
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
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
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
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
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
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), color_palette(int(cls_), bgr=True))

            # Draw bbox rectangle
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
            cv2.putText(im, f"{album_name}: {conf:.3f}", (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)

    # Mix image
    im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
    cv2.namedWindow("Name-Searcher", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Name-Searcher", scaled_size) 
    cv2.imshow("Name-Searcher", im)

def get_full_album_name(jsonstr):
    """Reads MusicBrainz JSON for album details."""
    j = json.loads(jsonstr)
    outstr = j["releases"][0]["title"] + " - " + j["releases"][0]["artist-credit"][0]["name"]
    try: 
        outstr += j["releases"][0]["artist-credit"][0]["joinphrase"] + " " + j["releases"][0]["artist-credit"][1]["name"]
    except KeyError as e:
        pass
    except IndexError as e:
        pass
    try: 
        outstr += j["releases"][0]["artist-credit"][1]["joinphrase"] + " " + j["releases"][0]["artist-credit"][2]["name"]
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
        lastjson = requests.get("https://musicbrainz.org/ws/2/release?query={}:\"{}\"&limit=1&fmt=json".format(request_type, catnum)).text
    return lastjson

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(growth_channels, growth_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(growth_channels, growth_channels, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(growth_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = self.relu3(self.conv3(out2))
        return x + self.scale * self.conv4(out3)

# RRDB (Residual in Residual Dense Block)
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.blocks = nn.Sequential(
            ResidualDenseBlock(in_channels, growth_channels),
            ResidualDenseBlock(in_channels, growth_channels),
            ResidualDenseBlock(in_channels, growth_channels),
        )

    def forward(self, x):
        return x + self.blocks(x)

# Generator
class ESRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_rrdb=23, growth_channels=32):
        super(ESRGANGenerator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(growth_channels) for _ in range(num_rrdb)])
        self.conv2 = nn.Conv2d(growth_channels, growth_channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(growth_channels, growth_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(growth_channels, growth_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )
        self.conv3 = nn.Conv2d(growth_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.rrdb_blocks(x1)
        x3 = self.conv2(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.conv3(x)


def upscale_image(image, model_path='weights/8x_NMKD-Typescale_175k.pth'):
    """
    Upscales an image using an ESRGAN model.

    Parameters:
        image (numpy.ndarray): Input image in OpenCV format (BGR).
        model_path (str): Path to the ESRGAN model file (.pth).

    Returns:
        numpy.ndarray: Upscaled image in OpenCV format (BGR).
    """
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ESRGAN model architecture
    model = ESRGANGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, pickle_module=pickle))
    model.eval()

    # Convert the OpenCV image to a tensor (normalize to [0, 1])
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass the image through the model
        output_tensor = model(img_tensor)
    
    # Convert the output tensor back to an image
    output_image = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_image = ToPILImage()(output_image)
    
    # Convert the PIL image back to OpenCV format
    output_image = np.array(output_image)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    return output_image


def scan_barcode(image):
    """Scans the image for barcodes."""
    # Convert the image to grayscale (ZBar works better with grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # # Get the dimensions of the image
    # height, width = gray_image.shape
    
    # # Define the size of the corner regions (1/4th of width and height)
    # corner_height, corner_width = height // 4, width // 4

    # # Define corner regions: top-left, top-right, bottom-left, bottom-right
    # corners = {
    #     "top_left": gray_image[:corner_height, :corner_width],
    #     "top_right": gray_image[:corner_height, -corner_width:],
    #     "bottom_left": gray_image[-corner_height:, :corner_width],
    #     "bottom_right": gray_image[-corner_height:, -corner_width:]
    # }
    
    # # Check each corner for barcodes
    # for corner_name, corner_image in corners.items():
    #     barcodes = decode(corner_image)
    #     if barcodes:
    #         # If a barcode is found, return its data
    #         return barcodes[0].data.decode('utf-8')
    
    # Use pyzbar to decode the barcode(s) in the image
    barcodes = decode(gray_image)
    
    if barcodes:
        # Return the first decoded barcode's data as a string
        return barcodes[0].data.decode('utf-8')
    else:
        # Return None if no barcode is detected
        return None
    
    # Return None if no barcode is detected in any corner
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

def scan_cropped_masked_image(im, segments, boxes, back_threshold=0.9, side_threshold=0.7, output_dir='cropped_images'):
    """Crops the image to the mask and scans it if confidence is greater than the threshold."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (mask, box) in enumerate(zip(segments, boxes)):
        conf = box[4]  # Confidence is the 5th element in the box array
        object_name = classes[int(box[5])] 
        # Create a binary mask for the current object
        binary_mask = mask.astype(np.uint8) * 255  # Convert mask to binary (0 or 255)

        # Ensure the mask is the same size as the original image
        mask_resized = cv2.resize(binary_mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(im, im, mask=mask_resized)

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_image = masked_image[y1:y2, x1:x2]

        # Save the cropped image
        output_path = os.path.join(output_dir, f'cropped_mask_{object_name}.png')
        cv2.imwrite(output_path, cropped_image)
        print(f'Saved cropped image: {output_path}')
        if conf > side_threshold and object_name == "Side":
            try:
                ocr_list = extract_text_with_paddleocr(cropped_image)
            except TypeError as e:
                return ""
            sorted_list = sorted(ocr_list, key=lambda s: sum(c.isdigit() for c in s) / len(s) if len(s) > 0 else 0)
            catnum = sorted_list[-1]
            if any(char.isdigit() for char in catnum):
                jsonstr = mb_request(catnum, "catno")
                try:
                    return get_full_album_name(jsonstr)
                except IndexError as e:
                    return catnum
            else:
                return catnum

        if conf > back_threshold and object_name == "Back":
            print(scan_barcode(cropped_image))

        if conf > back_threshold and object_name == "Front":
            print(scan_barcode(cropped_image))
        print(scan_barcode(cropped_image))
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
        img_processed, ratio, (pad_w, pad_h) = preprocess(img, model_height, model_width, ndtype)

        # Inference
        preds = session.run(None, {session.get_inputs()[0].name: img_processed})
        boxes, segments, _ = postprocess(preds, img, ratio, pad_w, pad_h, conf_threshold=args.conf, iou_threshold=args.iou)
        
        # Save cropped masked images if confidence is greater than 90%
        upscaled_img = upscale_image(img)
        name = scan_cropped_masked_image(upscaled_img, segments, boxes)

        # Draw and visualize results
        draw_and_visualize(upscaled_img, boxes, segments, name)

        # Break the loop on key press
        if cv2.waitKey(1) & 0xFF == ord("1"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
