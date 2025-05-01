import albumentations as A
from albumentations.pytorch import ToTensorV2 # Important for PyTorch
import torch  
import numpy as np 

# Define image size - choose based on model requirements or experiments
IMG_SIZE = 640 # Example size

def get_transforms(is_train=True):
    """Defines the transformations."""
    if is_train:
        # Training Transforms (with Augmentation)
        transform = A.Compose([
            # --- Augmentations ---
            A.HorizontalFlip(p=0.5), # Randomly flip horizontally
            # Add other augmentations here if desired (e.g., brightness, contrast)

            # --- Preprocessing ---
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0), # Resize
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0), # Normalize using ImageNet stats
            ToTensorV2(p=1.0) # Convert image and target to PyTorch tensors
        ], bbox_params=A.BboxParams(format='pascal_voc', # Your boxes are [x1, y1, x2, y2]
                                    min_area=1, # Ignore tiny boxes after transforms
                                    min_visibility=0.1, # Ignore boxes mostly out of view
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
        transformed = transform(image=np.array(image), # Convert PIL/OpenCV image to numpy array
                               bboxes=target['boxes'].tolist(),
                               labels=target['labels'].tolist())
        # Convert back to tensors and update target dict
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
        return transformed['image'], target # Return transformed image tensor and updated target dict

    return apply_transform # Return the function that applies the transforms

def collate_fn(batch):
    """
    Custom collate function for object detection batches.

    Args:
        batch: A list of tuples, where each tuple is (image, target)
               as returned by the LidarDataset's __getitem__.

    Returns:
        A tuple containing:
            - A tuple of all images in the batch.
            - A tuple of all target dictionaries in the batch.
    """
    return tuple(zip(*batch))