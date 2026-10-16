# Dual-Arm Robot Perception

The perception module of a collaborative dual-arm robot: two Franka Emika arms working together to cook and serve pasta. The full system had three modules. Task planning split each order into subtasks, motion planning executed them, and perception detected and located the items in the workspace (pans, strainers, pasta and sauce dispensers, the stovetop, the robots) so motion planning knew where they were.

I worked on the perception module alone, as a visiting researcher at the University of Waterloo from February to July 2020, as the final-year project of my B.Sc. Task planning and motion planning are not in this repository.

## Built on an open-source SSD300

The detector is Sagar Vinodababu's PyTorch implementation of SSD300 ([a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection), MIT License). The first commit in this repository (tag `upstream-9cad25b`) is that code, unmodified, at the version I started from. Everything after it is my work, so this shows exactly what I changed:

```
git diff upstream-9cad25b original-2020
```

## What I did

- **Detector setup and training.** Set up the SSD300 detector (VGG-16 base pretrained on ImageNet), trained it on a laptop, and updated its non-maximum suppression step to work with newer PyTorch versions.
- **Multi-dataset data pipeline.** Extended the data pipeline so training lists could merge PASCAL VOC 2007/2012 with two more sources: a robot-kitchen dataset and KITTI driving frames.
- **Kitchen dataset preparation.** Collected images with a depth camera, hand-annotated 100 images of the robot kitchen in PASCAL VOC format (pans, strainers, pasta and sauce dispensers, robots, stovetop), and wrote the tooling to convert YOLO annotations to VOC, build training lists, and count objects per class (`gastro/`).
- **Detection on video and frame sequences.** Split video into frames, ran detection over whole folders, filtered out classes that can't appear in a scene (no sofas on a KITTI road), and reported the pixel center of each detection for localization (`detect.py`, `video_into_frames.py`).
- **ROS integration proof of concept.** A `classifier_node` publishes a MoveIt `Grasp` message on `/grasp` for the motion-planning module (`ros/publisher_trial.py`). The position is predefined, standing in for one the detector would provide. The 2020 report describes this link as a `/coordinates` topic carrying a label and x, y, z; this script is the version that survives.

The original June 2020 report is in [`docs/Perception_Module_Report.pdf`](docs/Perception_Module_Report.pdf).

## A note on the results in the report

The per-class average precision table in the report (about 77% mAP on the VOC 2007 test set) matches the results the upstream tutorial publishes for its pretrained checkpoint. Read it as a benchmark of the SSD300 implementation, not of a model trained in this repository.

## Status: what was not finished

The detector was never trained on the kitchen classes. The kitchen labels were prepared and wired into the data pipeline, but the label map was not extended beyond the 20 VOC classes, so the pipeline dropped every kitchen object. The fixed version now reports those labels instead of skipping them silently.

Finishing it would take three steps: add the kitchen classes to the label map in `utils.py` (and merge the spelling variants found in the labels, such as `stainer` and `strainer`), retrain, and evaluate on held-out kitchen images.

Also not included: the model checkpoint (too large for GitHub), the datasets, and the step that converted KITTI frames to VOC layout.

## Versions

- **`original-2020` (tag):** the code exactly as I left it in June 2020, including the hardcoded Windows paths and the bugs listed below.
- **`main`:** the same project with these fixes:
  - Detection computed box centers as `(xmin+ymin)/2` and `(xmax+ymax)/2` instead of `(xmin+xmax)/2` and `(ymin+ymax)/2`.
  - The kitchen training list wrote full paths with a missing separator where image IDs were expected.
  - The check meant to skip images with no usable objects never triggered.
  - The ROS publisher mixed tabs and spaces and imported a package it didn't use.
  - The YOLO-to-VOC converter wrote malformed XML tag names.
  - Hardcoded paths were replaced with command-line arguments, and the code now runs on current PyTorch and Pillow releases.

## Usage

```bash
pip install -r requirements.txt

# Build training and test lists (kitchen and KITTI sources are optional)
python create_data_lists.py --voc07 path/to/VOC2007 --voc12 path/to/VOC2012 \
    --gastro path/to/kitchen_dataset --kitti path/to/kitti_voc

python train.py   # trains SSD300; writes checkpoint_ssd300.pth.tar
python eval.py    # evaluates on the VOC 2007 test set

# Detection on one image, a folder of frames, or a video
python detect.py --image path/to/image.jpg
python detect.py --frames path/to/kitti/image_02/data --exclude kitti --save-dir out/
python detect.py --video path/to/video.mp4 --frames-out frames/ --save-dir out/

# Kitchen dataset tooling
python gastro/convert.py --labels yolo_labels/ --images images/ --output labels/
python gastro/generate_train.py --images images/ --annotations labels/ --output train.txt
python gastro/stats.py labels/
```

The ROS publisher needs a ROS workspace with `moveit_msgs` installed. Place the script in one of the workspace's packages, make it executable, and start it with `rosrun`.

## Acknowledgements

- Supervised by Dr. William Melek, University of Waterloo.
- SSD300 implementation by Sagar Vinodababu.
- SSD: Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016.
- Datasets: PASCAL VOC 2007 and 2012; KITTI (Geiger et al., Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago).

## License

MIT. See [LICENSE](LICENSE), which carries both the original author's copyright notice and mine.
