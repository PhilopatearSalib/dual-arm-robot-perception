from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import PIL
import torch
import cv2
from os.path import join, isfile
from os import listdir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/PyTorch_Object_Detection/For git/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
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
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        print("center pixel of", det_labels[i],"is :", (int)(box_location[0]+box_location[1])/2,",",(int)(box_location[2]+box_location[3])/2)
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw
    
    return annotated_image

def video_into_frames( video_path,saved_path ):

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(os.path.join(saved_path , "frame%d.jpg" % count), image) # save frame as JPEG file
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      count += 1
      

if __name__ == '__main__':
    video_path = 'D:\my college books\SEMESTERS\8 SENIOR2\Waterloo\cutting video into frames'  #put the video path in here
    saving_path = 'D:/my college books/SEMESTERS/8 SENIOR2/Spring20/Datasets/2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync/image_02/data' #put the desired path to save the frames in it
    video_into_frames (video_path,saving_path)
    imageObjects = []
    imgFiles  = [
        join( saving_path, fn )                    # Create full paths to images
        for fn in listdir( saving_path )           # For each item in the image folder
        if isfile( join( saving_path, fn ) )       # If the item is indeed a file
        and fn.lower().endswith(('.png','.jpg'))   # Which ends with an image suffix (can add more types here if needed)
    ]
    
    for imgFile in imgFiles:
        original_image = Image.open(imgFile, mode='r')
        original_image = original_image.convert('RGB')
        prohibted_list=['bottle','bird','tvmonitor','aeroplane', 'boat','chair', 'cat','cow','diningtable', 'horse', 'pottedplant', 'sheep','sofa',] #for the Kitti dataset we wouldn't face any of those classes 
        output_im = detect(original_image, min_score=0.2, max_overlap=0.49, top_k=200,suppress= prohibted_list).show()

    
    
#    img_path = 'D:/my college books/SEMESTERS/8 SENIOR2/Waterloo/PyTorch_Object_Detection/VOCdevkit/VOC2007/JPEGImages/000330.jpg'
#    original_image = Image.open(img_path, mode='r')
#    original_image = original_image.convert('RGB')
#    detect(original_image, min_score=0.2, max_overlap=0.46, top_k=200).show()