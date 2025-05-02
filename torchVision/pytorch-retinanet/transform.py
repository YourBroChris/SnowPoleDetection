import albumentations as A
from albumentations.pytorch import ToTensorV2 # Important for PyTorch
import torch
import numpy as np
import cv2 # Needed for border_mode in ShiftScaleRotate
import math

# Define image size - choose based on model requirements or experiments
IMG_SIZE = 1600 # Example size

def get_transforms(is_train=True):
    """Defines the transformations."""
    if is_train:
        # Training Transforms (with Augmentation)
        transform = A.Compose([
            # --- Geometric Augmentations ---
            # Keep HorizontalFlip
            A.HorizontalFlip(p=0.5),

            
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
                rotate=(-10, 10),
                shear={'x': (-10, 10), 'y': (-10, 10)},
                p=0.7,
                # Corrected parameter names:
                border_mode=cv2.BORDER_CONSTANT, # Use border_mode
                fill=0                         # Use fill for the constant value
            ),

            # --- Pixel-level Augmentations ---
            # Adjust brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2, # Max 20% change
                contrast_limit=0.2,   # Max 20% change
                p=0.5                # Apply 50% of the time
            ),

            # Adjust Hue, Saturation, Value
            A.HueSaturationValue(
                hue_shift_limit=10,   # Max 10 degree shift in hue
                sat_shift_limit=20,   # Max 20% change in saturation
                val_shift_limit=10,   # Max 10% change in value (brightness)
                p=0.3                # Apply 30% of the time
            ),

            # Add slight blur sometimes
            A.Blur(blur_limit=(3, 7), p=0.1), # Kernel size between 3 and 7, 10% chance

            # Add noise sometimes
            A.GaussNoise(
                p=0.1 # Keep original probability
            ),

            # --- Preprocessing ---
            # Resize MUST come after geometric transforms that change size/shape
            # but typically before normalization
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),

            # Normalize using ImageNet stats (usually done last before ToTensor)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),

            # Convert image and target to PyTorch tensors
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', # Your boxes are [x1, y1, x2, y2]
                                    min_area=1,          # Ignore tiny boxes after transforms
                                    min_visibility=0.1,  # Ignore boxes mostly out of view
                                    label_fields=['labels'] # Tells albumentations labels are associated with boxes
                                    ))
    else:
        # Validation/Test Transforms (Preprocessing only)
        transform = A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) # Still need bbox_params

    # This wrapper function is needed so the Dataset's __getitem__ can call it easily
    def apply_transform(image, target):
        # Albumentations expects labels alongside boxes in a dictionary
        # Ensure image is a NumPy array
        img_np = np.array(image)

        # Ensure target boxes and labels are lists
        bboxes_list = target['boxes'].tolist() if isinstance(target['boxes'], torch.Tensor) else list(target['boxes'])
        labels_list = target['labels'].tolist() if isinstance(target['labels'], torch.Tensor) else list(target['labels'])

        try:
            transformed = transform(image=img_np,
                                    bboxes=bboxes_list,
                                    labels=labels_list)

            # Convert back to tensors and update target dict
            # Handle cases where bboxes might become empty after transformation
            transformed_bboxes = transformed['bboxes']
            if not transformed_bboxes: # If list is empty
                 target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                 target['labels'] = torch.empty((0,), dtype=torch.int64)
            else:
                 target['boxes'] = torch.tensor(transformed_bboxes, dtype=torch.float32)
                 target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

            return transformed['image'], target # Return transformed image tensor and updated target dict

        except Exception as e:
            print(f"Error during transformation: {e}")
            print(f"Image shape: {img_np.shape}")
            print(f"Original bboxes: {bboxes_list}")
            print(f"Original labels: {labels_list}")
            # Return original image and target or handle error appropriately
            # For simplicity, let's return original data if transform fails, but ideally log this
            # Ensure image is tensorized if returning original
            # NOTE: This simple fallback might not be ideal if normalization is crucial.
            transform_fallback = A.Compose([
                 A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
                 ToTensorV2(p=1.0)
            ])
            processed_fallback = transform_fallback(image=img_np)
            target['boxes'] = torch.tensor(bboxes_list, dtype=torch.float32) # Use original
            target['labels'] = torch.tensor(labels_list, dtype=torch.int64) # Use original
            return processed_fallback['image'], target


    return apply_transform # Return the function that applies the transforms

def collate_fn(batch):
    """
    Custom collate function for object detection batches. Handles possibly empty targets.

    Args:
        batch: A list of tuples, where each tuple is (image, target)
               as returned by the LidarDataset's __getitem__.

    Returns:
        A tuple containing:
            - A tensor of all images in the batch (stacked).
            - A list of all target dictionaries in the batch.
    """
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    # Stack images if they have the same size (which they should after Resize)
    images = torch.stack(images, dim=0)

    return images, targets # Return stacked images and list of targets