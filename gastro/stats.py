"""
Count annotated objects per class in a folder of YOLO .txt or PASCAL VOC .xml annotation files.

Usage:
    python stats.py LABELS_DIR
"""
import argparse
import os
import xml.etree.ElementTree as ET
from collections import Counter

from convert import CLASS_MAPPING

parser = argparse.ArgumentParser(description='Count objects per class.')
parser.add_argument('labels_dir')
args = parser.parse_args()

count = Counter()
for filename in os.listdir(args.labels_dir):
    path = os.path.join(args.labels_dir, filename)
    if filename.endswith('.txt') and filename != 'classes.txt':
        with open(path) as file:
            for line in file:
                fields = line.split()
                if fields:
                    count[CLASS_MAPPING.get(fields[0], 'class ' + fields[0])] += 1
    elif filename.endswith('.xml'):
        for obj in ET.parse(path).getroot().iter('object'):
            count[obj.find('name').text.strip().lower()] += 1

for name, n in count.most_common():
    print('%-20s %d' % (name, n))
