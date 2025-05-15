import tensorflow as tf
import numpy as np

from config import ImageEncoderDecoderConfig

class ImageEncoder(tf.keras.Model):
    """Image encoder based on pretrained EfficientNet-Lite."""
    
    def __init__(self, config: ImageEncoderDecoderConfig):
        super().__init__()
        self.config = config
        
        # Initialize EfficientNet-Lite0 model (smallest variant)
        # Remove the top classification layer
        self.efficient_net = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(*config.image_size, config.channels)
        )
        
        # Freeze most of the model layers
        for layer in self.efficient_net.layers[:-10]:  # Freeze all but the last 10 layers
            layer.trainable = False
        
        # Global pooling layer
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        
        # Projection layers for mean and logvar
        self.mean_projection = tf.keras.layers.Dense(
            config.latent_dim,
            name="image_mean_projection"
        )
        self.logvar_projection = tf.keras.layers.Dense(
            config.latent_dim,
            name="image_logvar_projection"
        )
    
    def call(self, inputs, training=False):
        # Preprocess inputs to match EfficientNet requirements
        x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
        
        # Get EfficientNet features
        features = self.efficient_net(x, training=training)
        
        # Global pooling
        pooled = self.global_pooling(features)
        
        # Project to mean and logvar
        mean = self.mean_projection(pooled)
        logvar = self.logvar_projection(pooled)
        
        return mean, logvar
