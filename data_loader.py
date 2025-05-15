import tensorflow as tf
import numpy as np
from datasets import load_dataset
import cv2

def load_image_data(image_config, max_samples=10000):
    """Load and preprocess image dataset.
    
    Args:
        image_config: Configuration for image encoder/decoder
        max_samples: Maximum number of samples
        
    Returns:
        Training and validation image data
    """
    # We'll use CIFAR-100 for diversity and convenience
    print("Loading CIFAR-100 dataset...")
    cifar_dataset = load_dataset("cifar100")
    
    # Process images to desired size and format
    processed_images = []
    
    # Process train split
    for i, item in enumerate(cifar_dataset["train"]):
        img = item["img"]
        # Convert PIL to numpy and resize
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, image_config.image_size)
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        processed_images.append(img_normalized)
        
        if len(processed_images) >= max_samples:
            break
    
    # Convert to numpy array
    processed_images = np.array(processed_images)
    
    # Manually split into train and validation (90/10)
    num_samples = len(processed_images)
    num_val = int(num_samples * 0.1)
    
    # Shuffle the data
    indices = np.random.permutation(num_samples)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_images = processed_images[train_indices]
    val_images = processed_images[val_indices]
    
    print(f"Data loaded: {len(train_images)} training, {len(val_images)} validation samples")
    
    # Skip the Oxford Flowers dataset to avoid potential issues
    
    return train_images, val_images