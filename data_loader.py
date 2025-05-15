import tensorflow as tf
import numpy as np
from datasets import load_dataset
import cv2
from sklearn.model_selection import train_test_split

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
    
    # Split into train and validation
    train_images, val_images = train_test_split(processed_images, test_size=0.1, random_state=42)
    
    # Add more diversity by incorporating some higher-resolution images from a different dataset
    try:
        print("Loading Oxford Flowers dataset for additional image diversity...")
        flowers_dataset = load_dataset("oxford-flowers-102", split="train")
        
        additional_images = []
        for i, item in enumerate(flowers_dataset):
            if i >= len(val_images):  # Match validation size
                break
                
            img = item["image"]
            img_array = np.array(img)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:  # Handle RGBA
                img_array = img_array[:, :, :3]
                
            img_resized = cv2.resize(img_array, image_config.image_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            additional_images.append(img_normalized)
            
            if len(additional_images) >= len(val_images):
                break
                
        # Replace half of validation set with flowers for diversity
        if additional_images:
            replace_count = min(len(val_images) // 2, len(additional_images))
            val_images[:replace_count] = additional_images[:replace_count]
            
    except Exception as e:
        print(f"Error loading additional image dataset: {e}")
        print("Continuing with CIFAR-100 only")
    
    return np.array(train_images), np.array(val_images)
