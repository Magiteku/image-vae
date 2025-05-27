import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import random
from typing import Tuple, List, Dict, Optional, Union, Any

def load_image_data(image_config, source_type="tf_builtin", source="cifar100", max_samples=10000, val_split=0.1, **kwargs):
    """Unified image data loader supporting multiple data sources.
    
    Args:
        image_config: Configuration for image encoder/decoder
        source_type: Type of data source ('tf_builtin', 'tfds', 'directory', 'file_list', 'hub')
        source: Specific source (dataset name, directory path, file path)
        max_samples: Maximum number of samples to load
        val_split: Fraction of data to use for validation (if not predefined)
        **kwargs: Additional arguments specific to the source type
        
    Returns:
        Training and validation image data as numpy arrays
    """
    print(f"Loading image data from {source_type} source: {source}")
    
    if source_type == "tf_builtin":
        # Load from built-in TensorFlow datasets
        return _load_from_tf_builtin(image_config, source, max_samples)
    elif source_type == "tfds":
        # Load from TensorFlow Datasets
        try:
            import tensorflow_datasets as tfds
            return _load_from_tfds(image_config, source, max_samples)
        except ImportError:
            print("Warning: tensorflow_datasets not installed. Falling back to CIFAR-100.")
            return _load_from_tf_builtin(image_config, "cifar100", max_samples)
    elif source_type == "directory":
        # Load from a directory of images
        return _load_from_directory(image_config, source, max_samples, val_split, **kwargs)
    elif source_type == "file_list":
        # Load from a file containing paths to images
        return _load_from_file_list(image_config, source, max_samples, val_split, **kwargs)
    elif source_type == "hub" or source_type == "deeplake":
        # Load from Deep Lake (ActiveLoop) datasets
        return _load_from_hub(image_config, source, max_samples, val_split, **kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

def _preprocess_image(img, target_size, normalize=True):
    """Preprocess a single image for the VAE.
    
    Args:
        img: Input image (numpy array)
        target_size: Target size as (width, height)
        normalize: Whether to normalize pixel values to [0,1]
        
    Returns:
        Preprocessed image as numpy array
    """
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # Handle RGBA
        img = img[:, :, :3]
        
    # Resize
    img_resized = cv2.resize(img, target_size)
    
    # Normalize to [0, 1] if requested
    if normalize:
        img_normalized = img_resized.astype(np.float32) / 255.0
        return img_normalized
    
    return img_resized

def _load_from_hub(image_config, dataset_name, max_samples, val_split=0.1, **kwargs):
    """Load images from Hub (ActiveLoop) datasets with robust fallbacks.
    
    Args:
        image_config: Configuration containing image size
        dataset_name: Name of Hub dataset ('ffhq', etc.)
        max_samples: Maximum number of samples to load
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_images, val_images) as numpy arrays
    """
    if dataset_name.lower() == "ffhq":
        return load_ffhq_with_fallbacks(image_config, max_samples, val_split)
    else:
        # For other datasets, use the basic hub loading
        try:
            import hub
            print(f"Loading {dataset_name} dataset from Hub...")
            ds = hub.load(f"hub://activeloop/{dataset_name}")
            # ... rest of the loading logic would go here
        except Exception as e:
            print(f"Error loading {dataset_name} from Hub: {e}")
            print("Falling back to CIFAR-100...")
            return _load_from_tf_builtin(image_config, "cifar100", max_samples)

def _load_from_hub(image_config, dataset_name, max_samples, val_split=0.1, **kwargs):
    """Load images from Deep Lake (ActiveLoop) datasets.
    
    Args:
        image_config: Configuration containing image size
        dataset_name: Name of Deep Lake dataset ('ffhq', etc.)
        max_samples: Maximum number of samples to load
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_images, val_images) as numpy arrays
    """
    if dataset_name.lower() == "ffhq":
        return load_ffhq_with_deeplake(image_config, max_samples, val_split)
    else:
        # For other datasets, fall back to CIFAR-100
        print(f"Dataset {dataset_name} not supported yet. Falling back to CIFAR-100...")
        return _load_from_tf_builtin(image_config, "cifar100", max_samples)

def load_ffhq_with_deeplake(image_config, max_samples=10000, val_split=0.1):
    """Load FFHQ using Deep Lake (the correct ActiveLoop method)."""
    
    print("🔄 Loading FFHQ dataset using Deep Lake...")
    
    try:
        import deeplake
        print("✅ Deep Lake library imported successfully")
        
        # Load FFHQ dataset from Deep Lake
        print("📦 Loading FFHQ from hub://activeloop/ffhq...")
        ds = deeplake.load("hub://activeloop/ffhq")
        
        print(f"✅ FFHQ dataset loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   Total samples: {len(ds)}")
        print(f"   Available tensors: {list(ds.tensors.keys())}")
        
        # Choose appropriate image resolution based on target size
        target_size = image_config.image_size[0]  # Assuming square images
        
        if target_size <= 128:
            # Use 128x128 images
            image_tensor = ds.images_128.image
            print(f"🎯 Using 128x128 images (target: {target_size}x{target_size})")
        else:
            # Use 1024x1024 images
            image_tensor = ds.images_1024.image
            print(f"🎯 Using 1024x1024 images (target: {target_size}x{target_size})")
        
        # Limit samples
        total_samples = min(len(image_tensor), max_samples)
        print(f"🔢 Processing {total_samples} samples...")
        
        # Create train/validation split (first 60k for train, rest for val)
        if total_samples > 60000:
            train_end = 60000
            val_start = 60000
        else:
            train_end = int(total_samples * (1 - val_split))
            val_start = train_end
        
        # Process training images
        print("🏋️ Processing training images...")
        train_images = []
        for i in range(min(train_end, total_samples)):
            if i % 1000 == 0:
                print(f"   Processed {i}/{min(train_end, total_samples)} training images...")
            
            try:
                # Get image as numpy array
                img = image_tensor[i].numpy()
                
                # Preprocess image
                processed_img = _preprocess_image(img, image_config.image_size)
                train_images.append(processed_img)
                
            except Exception as e:
                print(f"⚠️ Error processing training image {i}: {e}")
                continue
        
        # Process validation images
        print("🧪 Processing validation images...")
        val_images = []
        val_end = min(total_samples, val_start + int(total_samples * val_split))
        
        for i in range(val_start, val_end):
            if (i - val_start) % 500 == 0:
                print(f"   Processed {i - val_start}/{val_end - val_start} validation images...")
            
            try:
                # Get image as numpy array
                img = image_tensor[i].numpy()
                
                # Preprocess image
                processed_img = _preprocess_image(img, image_config.image_size)
                val_images.append(processed_img)
                
            except Exception as e:
                print(f"⚠️ Error processing validation image {i}: {e}")
                continue
        
        # Convert to numpy arrays
        train_images = np.array(train_images)
        val_images = np.array(val_images)
        
        print(f"✅ FFHQ loading completed successfully!")
        print(f"📊 Final dataset statistics:")
        print(f"   Training samples: {len(train_images)}")
        print(f"   Validation samples: {len(val_images)}")
        print(f"   Training shape: {train_images.shape}")
        print(f"   Validation shape: {val_images.shape}")
        
        return train_images, val_images
        
    except ImportError:
        print("❌ Deep Lake library not installed")
        print("💡 Install with: pip install deeplake")
        print("🔄 Falling back to CelebA dataset...")
        return load_celeba_fallback(image_config, max_samples, val_split)
    
    except Exception as e:
        print(f"❌ Error loading FFHQ with Deep Lake: {e}")
        print("🔄 Falling back to CelebA dataset...")
        return load_celeba_fallback(image_config, max_samples, val_split)

def load_celeba_fallback(image_config, max_samples, val_split):
    """Load CelebA as a high-quality face dataset fallback."""
    try:
        print("\n📦 Loading CelebA dataset as FFHQ alternative...")
        import tensorflow_datasets as tfds
        
        ds = tfds.load('celeb_a', split='train', as_supervised=False)
        
        print("✅ CelebA loaded successfully")
        
        processed_images = []
        count = 0
        
        for sample in ds:
            if count >= max_samples:
                break
                
            if count % 1000 == 0:
                print(f"   Processed {count}/{max_samples} CelebA images...")
            
            try:
                img = sample['image'].numpy()
                processed_img = _preprocess_image(img, image_config.image_size)
                processed_images.append(processed_img)
                count += 1
            except Exception:
                continue
        
        if len(processed_images) > 0:
            # Split data
            num_samples = len(processed_images)
            num_val = int(num_samples * val_split)
            
            indices = np.random.permutation(num_samples)
            train_indices = indices[num_val:]
            val_indices = indices[:num_val]
            
            train_images = np.array([processed_images[i] for i in train_indices])
            val_images = np.array([processed_images[i] for i in val_indices])
            
            print("✅ CelebA loaded as FFHQ alternative")
            return train_images, val_images
        
    except Exception as e:
        print(f"❌ CelebA fallback also failed: {e}")
    
    # Final fallback to CIFAR-100
    print("\n⚠️ All face dataset methods failed. Using CIFAR-100.")
    print("   Note: This will not produce good face generation results.")
    
    return _load_from_tf_builtin(image_config, "cifar100", max_samples)

# Alias for backward compatibility
def load_ffhq_with_fallbacks(image_config, max_samples=10000, val_split=0.1):
    """Alias for load_ffhq_with_deeplake - kept for backward compatibility."""
    return load_ffhq_with_deeplake(image_config, max_samples, val_split)

def _load_from_tf_builtin(image_config, dataset_name, max_samples):
    """Load images from TensorFlow's built-in datasets.
    
    Args:
        image_config: Configuration containing image size
        dataset_name: Name of built-in dataset ('cifar10', 'cifar100', 'mnist', etc.)
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (train_images, val_images) as numpy arrays
    """
    if dataset_name.lower() == "cifar10":
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name.lower() == "cifar100":
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar100.load_data()
    elif dataset_name.lower() == "mnist":
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    elif dataset_name.lower() == "fashion_mnist":
        (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown built-in dataset: {dataset_name}")
    
    # Limit training samples
    train_images = train_images[:max_samples]
    
    # Use 10% of test data for validation
    val_images = test_images[:len(test_images) // 10]
    
    print(f"Original shapes: {train_images.shape}, {val_images.shape}")
    
    # Process images
    processed_train = []
    processed_val = []
    
    for img in train_images:
        processed_train.append(_preprocess_image(img, image_config.image_size))
    
    for img in val_images:
        processed_val.append(_preprocess_image(img, image_config.image_size))
    
    train_data = np.array(processed_train)
    val_data = np.array(processed_val)
    
    print(f"Data loaded: {len(train_data)} training, {len(val_data)} validation samples")
    print(f"Processed shapes: {train_data.shape}, {val_data.shape}")
    
    return train_data, val_data

def _load_from_tfds(image_config, dataset_name, max_samples):
    """Load images from TensorFlow Datasets.
    
    Args:
        image_config: Configuration containing image size
        dataset_name: Name of TFDS dataset
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (train_images, val_images) as numpy arrays
    """
    import tensorflow_datasets as tfds
    
    # Load dataset
    try:
        train_ds, val_ds = tfds.load(
            dataset_name,
            split=['train', 'validation'],
            as_supervised=True,
            with_info=False
        )
    except (tfds.core.utils.DatasetNotFoundError, ValueError):
        try:
            # Some datasets use test instead of validation
            train_ds, val_ds = tfds.load(
                dataset_name,
                split=['train', 'test'],
                as_supervised=True,
                with_info=False
            )
        except:
            # Final fallback: use train with 90/10 split
            full_ds = tfds.load(
                dataset_name,
                split='train',
                as_supervised=True,
                with_info=False
            )
            train_size = int(0.9 * tfds.load_info(dataset_name).splits['train'].num_examples)
            train_ds = full_ds.take(train_size)
            val_ds = full_ds.skip(train_size)
    
    # Process datasets
    processed_train = []
    for img, _ in train_ds:
        if len(processed_train) >= max_samples:
            break
        try:
            img_array = img.numpy()
            processed_img = _preprocess_image(img_array, image_config.image_size)
            processed_train.append(processed_img)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    # For validation, take 10% of max_samples or at least 100
    val_samples = max(int(max_samples * 0.1), 100)
    processed_val = []
    for img, _ in val_ds:
        if len(processed_val) >= val_samples:
            break
        try:
            img_array = img.numpy()
            processed_img = _preprocess_image(img_array, image_config.image_size)
            processed_val.append(processed_img)
        except Exception as e:
            print(f"Error processing validation image: {e}")
            continue
    
    train_data = np.array(processed_train)
    val_data = np.array(processed_val)
    
    print(f"Data loaded: {len(train_data)} training, {len(val_data)} validation samples")
    print(f"Processed shapes: {train_data.shape}, {val_data.shape}")
    
    return train_data, val_data

def _load_from_directory(image_config, directory_path, max_samples, val_split=0.1, 
                         extensions=None, recursive=True, class_mode=None):
    """Load images from a directory structure.
    
    Args:
        image_config: Configuration containing image size
        directory_path: Path to directory containing images
        max_samples: Maximum number of samples to load
        val_split: Fraction of data to use for validation
        extensions: List of valid file extensions (default: ['.jpg', '.jpeg', '.png'])
        recursive: Whether to search subdirectories recursively
        class_mode: How to handle class subdirectories (None, 'categorical', etc.)
        
    Returns:
        Tuple of (train_images, val_images) as numpy arrays
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Convert extensions to lowercase if they're not already
    extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in extensions]
    
    # Find all image files
    image_paths = []
    if recursive:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            image_paths.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))
    
    print(f"Found {len(image_paths)} images in {directory_path}")
    
    # Limit to max_samples and shuffle
    if len(image_paths) > max_samples:
        random.shuffle(image_paths)
        image_paths = image_paths[:max_samples]
    
    # Load and process images
    processed_images = []
    for img_path in image_paths:
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            processed_img = _preprocess_image(img, image_config.image_size)
            processed_images.append(processed_img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    # Split into train and validation sets
    num_samples = len(processed_images)
    num_val = int(num_samples * val_split)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_images = np.array([processed_images[i] for i in train_indices])
    val_images = np.array([processed_images[i] for i in val_indices])
    
    print(f"Data loaded: {len(train_images)} training, {len(val_images)} validation samples")
    print(f"Processed shapes: {train_images.shape}, {val_images.shape}")
    
    return train_images, val_images

def _load_from_file_list(image_config, file_path, max_samples, val_split=0.1, base_dir=""):
    """Load images from a file containing paths to images.
    
    Args:
        image_config: Configuration containing image size
        file_path: Path to file listing image paths (one per line)
        max_samples: Maximum number of samples to load
        val_split: Fraction of data to use for validation
        base_dir: Base directory to prepend to relative paths in the file
        
    Returns:
        Tuple of (train_images, val_images) as numpy arrays
    """
    # Read file paths
    with open(file_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    # Filter out empty lines and comments
    image_paths = [path for path in image_paths if path and not path.startswith('#')]
    
    print(f"Found {len(image_paths)} image paths in {file_path}")
    
    # Limit to max_samples and shuffle
    if len(image_paths) > max_samples:
        random.shuffle(image_paths)
        image_paths = image_paths[:max_samples]
    
    # Load and process images
    processed_images = []
    for img_path in image_paths:
        # Handle base directory
        if base_dir and not os.path.isabs(img_path):
            img_path = os.path.join(base_dir, img_path)
            
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            processed_img = _preprocess_image(img, image_config.image_size)
            processed_images.append(processed_img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    # Split into train and validation sets
    num_samples = len(processed_images)
    num_val = int(num_samples * val_split)
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_images = np.array([processed_images[i] for i in train_indices])
    val_images = np.array([processed_images[i] for i in val_indices])
    
    print(f"Data loaded: {len(train_images)} training, {len(val_images)} validation samples")
    print(f"Processed shapes: {train_images.shape}, {val_images.shape}")
    
    return train_images, val_images