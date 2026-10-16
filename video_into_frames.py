import argparse
import os

import cv2


def video_into_frames(video_path, saved_path):
    """
    Split a video into JPEG frames (frame0.jpg, frame1.jpg, ...) in saved_path.

    :return: number of frames written
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError('Video not found: %s' % video_path)
    os.makedirs(saved_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(saved_path, "frame%d.jpg" % count), image)  # save frame as JPEG file
        count += 1
        success, image = vidcap.read()
    vidcap.release()
    print('Wrote %d frames to %s' % (count, saved_path))
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a video into frames.')
    parser.add_argument('video')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    video_into_frames(args.video, args.output_folder)
