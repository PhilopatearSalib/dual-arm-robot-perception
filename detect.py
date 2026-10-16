import argparse
import os

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch

from video_into_frames import video_into_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None  # loaded in load_model()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Classes that cannot appear in KITTI road scenes; used with --exclude kitti
KITTI_EXCLUDED = ['bottle', 'bird', 'tvmonitor', 'aeroplane', 'boat', 'chair', 'cat', 'cow', 'diningtable', 'horse',
                  'pottedplant', 'sheep', 'sofa']


def load_model(checkpoint_path):
    global model
    checkpoint = load_checkpoint(checkpoint_path)
    print('\nLoaded checkpoint from epoch %d.\n' % (checkpoint['epoch'] + 1))
    model = checkpoint['model'].to(device)
    model.eval()


def box_center(box):
    """
    Pixel center of a box given as [x_min, y_min, x_max, y_max].
    """
    return (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.


def _load_font():
    try:
        return ImageFont.truetype("./calibril.ttf", 15)
    except OSError:
        return ImageFont.load_default()


def _text_size(draw, font, text):
    if hasattr(draw, 'textbbox'):  # Pillow >= 8
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return font.getsize(text)


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image (a PIL Image) and a list of (label, (center_x, center_y)) detections
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image, []

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = _load_font()
    detections = []

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        center = box_center(box_location)
        detections.append((det_labels[i], center))
        print("center pixel of %s: (%d, %d)" % (det_labels[i], center[0], center[1]))
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = _text_size(draw, font, det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image, detections


def _image_files(folder):
    return sorted(os.path.join(folder, fn) for fn in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, fn)) and fn.lower().endswith(('.png', '.jpg', '.jpeg')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SSD300 detection on an image, a folder of frames, or a video.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', help='path to one image')
    source.add_argument('--frames', help='folder of images (for example, KITTI frames)')
    source.add_argument('--video', help='video file; it is split into frames first (see --frames-out)')
    parser.add_argument('--frames-out', default='./frames', help='where to write frames when using --video')
    parser.add_argument('--checkpoint', default='./checkpoint_ssd300.pth.tar')
    parser.add_argument('--exclude', nargs='*', default=None,
                        help="class names to ignore, or 'kitti' for the classes that cannot appear on a road")
    parser.add_argument('--min-score', type=float, default=0.2)
    parser.add_argument('--max-overlap', type=float, default=0.49)
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--save-dir', default=None, help='save annotated images here instead of opening them')
    args = parser.parse_args()

    suppress = args.exclude
    if suppress == ['kitti']:
        suppress = KITTI_EXCLUDED

    load_model(args.checkpoint)

    if args.image:
        files = [args.image]
    elif args.frames:
        files = _image_files(args.frames)
    else:
        video_into_frames(args.video, args.frames_out)
        files = _image_files(args.frames_out)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for path in files:
        original_image = Image.open(path, mode='r').convert('RGB')
        annotated, _ = detect(original_image, min_score=args.min_score, max_overlap=args.max_overlap,
                              top_k=args.top_k, suppress=suppress)
        if args.save_dir:
            annotated.save(os.path.join(args.save_dir, os.path.basename(path)))
        else:
            annotated.show()
