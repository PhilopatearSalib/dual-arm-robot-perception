"""
Write train.txt: one image ID (file name without extension) per line, as create_data_lists() expects.

Usage:
    python generate_train.py --images IMAGES_DIR --output train.txt [--annotations ANNOTATIONS_DIR]

With --annotations, only images that have a matching .xml annotation are listed.
"""
import argparse
import os

parser = argparse.ArgumentParser(description='Generate the train.txt image ID list.')
parser.add_argument('--images', required=True)
parser.add_argument('--output', default='train.txt')
parser.add_argument('--annotations', default=None)
args = parser.parse_args()

ids = []
for filename in sorted(os.listdir(args.images)):
    stem, ext = os.path.splitext(filename)
    if ext.lower() != '.jpg':
        continue
    if args.annotations and not os.path.isfile(os.path.join(args.annotations, stem + '.xml')):
        continue
    ids.append(stem)

with open(args.output, 'w') as f:
    f.write('\n'.join(ids) + '\n')
print('Wrote %d image IDs to %s' % (len(ids), args.output))
