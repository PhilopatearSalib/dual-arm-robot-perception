"""
Convert YOLO annotations (.txt, one "class x_center y_center width height" line per object,
normalized to the image size) into PASCAL VOC XML files.

Usage:
    python convert.py --labels LABELS_DIR --images IMAGES_DIR --output OUTPUT_DIR
"""
import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image

CLASS_MAPPING = {
    '0': 'robot',
    '1': 'pan',
    '2': 'pasta',
    '3': 'boiler',
    '4': 'strainer',
    '5': 'sauce dispenser',
    '6': 'pasta dispenser',
    '7': 'stovetop',
    '8': 'panhandle'
}


def create_root(file_prefix, width, height):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        # VOC boxes are integer pixel coordinates
        ET.SubElement(bbox, "xmin").text = str(int(round(voc_label[1])))
        ET.SubElement(bbox, "ymin").text = str(int(round(voc_label[2])))
        ET.SubElement(bbox, "xmax").text = str(int(round(voc_label[3])))
        ET.SubElement(bbox, "ymax").text = str(int(round(voc_label[4])))
    return root


def create_file(output_dir, file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write(os.path.join(output_dir, "{}.xml".format(file_prefix)))


def read_file(labels_dir, images_dir, output_dir, filename):
    file_prefix = os.path.splitext(filename)[0]
    img = Image.open(os.path.join(images_dir, "{}.jpg".format(file_prefix)))
    w, h = img.size
    with open(os.path.join(labels_dir, filename), 'r') as file:
        voc_labels = []
        for line in file:
            data = line.split()
            if len(data) != 5:
                continue
            if data[0] not in CLASS_MAPPING:
                print("Unknown class id {} in {}".format(data[0], filename))
                continue
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            voc_labels.append([CLASS_MAPPING[data[0]],
                               max(center_x - (bbox_width / 2), 0),
                               max(center_y - (bbox_height / 2), 0),
                               min(center_x + (bbox_width / 2), w),
                               min(center_y + (bbox_height / 2), h)])
        create_file(output_dir, file_prefix, w, h, voc_labels)
    print("Processing complete for file: {}".format(filename))


def start(labels_dir, images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in sorted(os.listdir(labels_dir)):
        if filename.endswith('.txt') and filename != 'classes.txt':
            read_file(labels_dir, images_dir, output_dir, filename)
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert YOLO annotations to PASCAL VOC XML.')
    parser.add_argument('--labels', required=True, help='folder of YOLO .txt files')
    parser.add_argument('--images', required=True, help='folder of the matching .jpg images')
    parser.add_argument('--output', required=True, help='folder for the VOC .xml files')
    args = parser.parse_args()
    start(args.labels, args.images, args.output)
