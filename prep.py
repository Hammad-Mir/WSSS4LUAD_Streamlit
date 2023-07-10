import cv2
import torch
import numpy as np
from torchvision import transforms, utils

def preprocess_image(img: np.ndarray, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resized = (224, 224)) -> torch.Tensor:
  
  preprocessing = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      transforms.Normalize(mean, std),
                                      ])
  return preprocessing(img.copy()).unsqueeze(0)

def mask_img_to_mask(mask_path, bg_path):
    
    gt_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
    gt_bg   = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2GRAY)
    
    tumor = (gt_mask == 52).astype(np.uint8).reshape(gt_mask.shape[0], gt_mask.shape[1], 1)
    stroma = (gt_mask == 94).astype(np.uint8).reshape(gt_mask.shape[0], gt_mask.shape[1], 1)
    normal = (gt_mask == 162).astype(np.uint8).reshape(gt_mask.shape[0], gt_mask.shape[1], 1)
    bg = (gt_bg/255).astype(np.uint8).reshape(gt_mask.shape[0], gt_mask.shape[1], 1)
    
    mask = np.concatenate((tumor, stroma, normal, bg), axis=2)
    
    return mask

def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    ) -> list[list[int]]:
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.

    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format
    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def img_resize(img, mask, factor):
    
    img_height, img_width, channels = img.shape
    
    #print(img.shape)
    
    new_height = int(factor * np.round(img_height/factor))
    new_width = int(factor * np.round(img_width/factor))
    
    img = cv2.resize(img, (max(new_width, 1), max(new_height, 1)))
    mask = cv2.resize(mask, (max(new_width, 1), max(new_height, 1)))
    
    #print(img.shape)
    
    slice_boxes = calculate_slice_bboxes(new_height, new_width, factor, factor, 0, 0)
    
    return img, mask, slice_boxes