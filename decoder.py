import tensorflow as tf
import numpy as np

from config import ImageEncoderDecoderConfig

class ImageDecoder(tf.keras.Model):
    """CNN-based image decoder with architecture mirroring EfficientNet-Lite."""
    
    def __init__(self, config: ImageEncoderDecoderConfig):
        super().__init__()
        self.config = config
        
        # Get encoder output dimensions
        # For EfficientNetB0, feature dimensions are significantly reduced
        # Typical dimensions would be 7x7 with 1280 filters for 224x224 input
        self.min_h = self.config.image_size[0] // 32
        self.min_w = self.config.image_size[1] // 32
        self.feature_dim = 1280  # EfficientNetB0 final feature dimension
        
        # Initial dense projection from latent space to feature map
        self.latent_dense = tf.keras.layers.Dense(
            self.min_h * self.min_w * self.feature_dim // 4,  # Reduced dimension for efficiency
            name="latent_dense"
        )
        self.reshape = tf.keras.layers.Reshape((
            self.min_h,
            self.min_w,
            self.feature_dim // 4
        ))
        
        # Transposed convolution blocks for upsampling
        # We'll use 5 upsampling blocks to go from 7x7 to 224x224 (5 doublings: 7→14→28→56→112→224)
        self.upsampling_blocks = []
        filter_dims = [self.feature_dim // 4, 256, 128, 64, 32]
        
        for i in range(5):
            self.upsampling_blocks.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2DTranspose(
                        filters=filter_dims[min(i, len(filter_dims)-1)],
                        kernel_size=3,
                        strides=2,
                        padding="same",
                        activation=None,
                        name=f"image_upconv_{i}"
                    ),
                    tf.keras.layers.BatchNormalization(name=f"image_upbn_{i}"),
                    tf.keras.layers.LeakyReLU(alpha=0.2, name=f"image_upleaky_{i}")
                ])
            )
            
        # Final output layer
        self.output_layer = tf.keras.layers.Conv2D(
            filters=self.config.channels,
            kernel_size=3,
            padding="same",
            activation="sigmoid",
            name="image_output"
        )
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Project to initial feature map
        x = self.latent_dense(x)
        x = self.reshape(x)
        
        # Apply upsampling blocks
        for block in self.upsampling_blocks:
            x = block(x, training=training)
            
        # Final output layer
        x = self.output_layer(x)
        
        return x
