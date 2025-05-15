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

def create_data_generators(image_config, batch_size=32, max_samples=5000):
    """Create data generators for training the image VAE.
    
    Args:
        image_config: Image configuration
        batch_size: Batch size
        max_samples: Maximum number of samples
        
    Returns:
        Training and validation data generators
    """
    from data_loader import load_image_data
    
    # Load image data
    train_images, val_images = load_image_data(image_config, max_samples=max_samples)
    
    # Create data generators
    train_generator = ImageDataGenerator(train_images, batch_size, shuffle=True)
    val_generator = ImageDataGenerator(val_images, batch_size, shuffle=False)
    
    return train_generator, val_generator
