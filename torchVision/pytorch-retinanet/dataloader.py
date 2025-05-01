import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image # Or import cv2 if using OpenCV
import os

class LidarDataset(Dataset):
    """Custom Dataset for loading Lidar image data with CSV annotations."""

    def __init__(self, annotations_file, classes_file, img_dir, transforms=None):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations
                                       (e.g., 'path/to/lidar_train_annotations.csv').
            classes_file (string): Path to the csv file with class-to-id mapping
                                   (e.g., 'path/to/classes.csv').
            img_dir (string): Directory with all the images
                              (e.g., '.' if paths in csv are relative to project root,
                               or '/path/to/retinanet_data/lidar/images/train' if csv only has image name).
                              Adjust based on how paths are stored in your annotations_file.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(annotations_file, header=None, names=['img_path', 'x1', 'y1', 'x2', 'y2', 'class_name'])
        self.img_dir = img_dir
        self.transforms = transforms

        # Create class name to id mapping
        classes_df = pd.read_csv(classes_file, header=None, names=['name', 'id'])
        self.class_to_id = {row['name']: row['id'] for index, row in classes_df.iterrows()}
        # Add background class explicitly? Torchvision models often assume background is class 0,
        # so if your classes.csv starts IDs from 0, you might need to adjust IDs (+1)
        # or carefully configure the model head. Let's assume classes.csv starts at 0 for now.
        # Check RetinaNet documentation for specifics on background class handling.

        # Group annotations by image path and get unique image paths
        self.image_paths = self.annotations['img_path'].unique().tolist()

        # Pre-group annotations for faster lookup in __getitem__
        self.img_annotations = self.annotations.groupby('img_path').agg(list)
        # Convert coordinate columns from lists of strings/numbers to lists of floats/ints
        for col in ['x1', 'y1', 'x2', 'y2']:
             self.img_annotations[col] = self.img_annotations[col].apply(lambda x: [float(i) for i in x])


    def __len__(self):
        """Returns the total number of unique images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the image and its corresponding annotations for a given index.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            tuple: (image, target) where target is a dictionary containing
                   'boxes' and 'labels'.
        """
        # Get image path
        img_path_relative = self.image_paths[idx]
        # Construct full image path - Adjust if paths in CSV are absolute vs relative
        
        full_img_path = img_path_relative # Use the path directly from the CSV

        # Load image
        # If using PIL:
        image = Image.open(full_img_path).convert("RGB")


        # Get annotations for this image
        img_annots = self.img_annotations.loc[img_path_relative]
        boxes = []
        labels = []

        for i in range(len(img_annots['class_name'])):
            x1 = img_annots['x1'][i]
            y1 = img_annots['y1'][i]
            x2 = img_annots['x2'][i]
            y2 = img_annots['y2'][i]
            class_name = img_annots['class_name'][i]

            # Ensure coordinates are valid (x1 < x2, y1 < y2) - add checks if needed
            # if x1 >= x2 or y1 >= y2: continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[class_name])

        # Convert annotations to Tensors
        # Make sure dtype is correct (Float for boxes, Int64 for labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create the target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # Torchvision models might also require "image_id", "area", "iscrowd"
        # You might need to add these based on specific model requirements/training pipeline
        target["image_id"] = torch.tensor([idx])
        # Calculate area - important if using COCO metrics
        if boxes.shape[0] > 0 :
             area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
              # handle case with no boxes
              area = torch.tensor([], dtype=torch.float32)

        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64) # Assume no crowd regions

        # Apply transformations
        if self.transforms:
            # Standard Torchvision transforms usually take (image, target)
            # Ensure your transform pipeline handles both correctly, especially
            # for spatial transforms that affect bounding boxes.
            image, target = self.transforms(image, target) # Check specific transform library docs

        return image, target
