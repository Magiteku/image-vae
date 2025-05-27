import tensorflow as tf
import numpy as np

class ImageDataGenerator(tf.keras.utils.Sequence):
    """Data generator for image VAE training."""
    
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.indices = np.arange(self.n_samples)
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:min((idx + 1) * self.batch_size, self.n_samples)]
        
        # Create batch data
        batch_data = np.array([self.data[i] for i in batch_indices])
        
        # For VAE, inputs = targets
        return batch_data, batch_data
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_data_generators(image_config, batch_size=32, max_samples=5000, dataset_type="ffhq"):
    """Create data generators for training the image VAE.
    
    Args:
        image_config: Image configuration
        batch_size: Batch size
        max_samples: Maximum number of samples
        dataset_type: Type of dataset to load ('ffhq', 'cifar100', etc.)
        
    Returns:
        Training and validation data generators
    """
    from data_loader import load_image_data
    
    # Load image data based on dataset type
    if dataset_type.lower() == "ffhq":
        # Load FFHQ dataset from Deep Lake
        train_images, val_images = load_image_data(
            image_config, 
            source_type="deeplake", 
            source="ffhq", 
            max_samples=max_samples
        )
    elif dataset_type.lower() in ["cifar10", "cifar100", "mnist", "fashion_mnist"]:
        # Load from TensorFlow built-in datasets
        train_images, val_images = load_image_data(
            image_config, 
            source_type="tf_builtin", 
            source=dataset_type, 
            max_samples=max_samples
        )
    else:
        # Default fallback to CIFAR-100
        print(f"Unknown dataset type '{dataset_type}', falling back to CIFAR-100")
        train_images, val_images = load_image_data(
            image_config, 
            source_type="tf_builtin", 
            source="cifar100", 
            max_samples=max_samples
        )
    
    # Create data generators
    train_generator = ImageDataGenerator(train_images, batch_size, shuffle=True)
    val_generator = ImageDataGenerator(val_images, batch_size, shuffle=False)
    
    print(f"Created data generators:")
    print(f"  Training batches: {len(train_generator)}")
    print(f"  Validation batches: {len(val_generator)}")
    
    return train_generator, val_generator