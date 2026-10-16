import argparse
from utils import create_data_lists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the training and test data lists (JSON) for SSD300.')
    parser.add_argument('--voc07', required=True, help="path to the 'VOC2007' folder")
    parser.add_argument('--voc12', required=True, help="path to the 'VOC2012' folder")
    parser.add_argument('--output', default='./', help='folder where the JSON files are saved')
    parser.add_argument('--gastro', default=None,
                        help="optional robot-kitchen dataset folder (with 'images/' and 'train.txt')")
    parser.add_argument('--gastro-annotations', default='labels',
                        help='name of the VOC-format annotation folder inside the kitchen dataset folder')
    parser.add_argument('--kitti', default=None, help='optional folder of KITTI frames in VOC layout')
    args = parser.parse_args()

    create_data_lists(voc07_path=args.voc07,
                      voc12_path=args.voc12,
                      output_folder=args.output,
                      gastro_path=args.gastro,
                      gastro_annotations=args.gastro_annotations,
                      kitti_path=args.kitti)
