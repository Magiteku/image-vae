import tensorflow as tf
import numpy as np

from config import BottleneckConfig

class MultiLayerBottleneck(tf.keras.layers.Layer):
    """Multi-layer bottleneck for VAE with complex-valued support."""
    
    def __init__(self, config: BottleneckConfig):
        super().__init__()
        self.config = config
        self.dim = config.latent_dim
        self.use_complex = config.use_complex
        
        # Define activation function
        if config.activation == "leaky_relu":
            self.activation_fn = lambda x: tf.keras.layers.LeakyReLU(
                negative_slope=config.leaky_alpha  # Updated from alpha to negative_slope
            )(x)
        else:
            self.activation_fn = lambda x: tf.keras.activations.get(config.activation)(x)
        
        # Define layers
        self.linear1 = tf.keras.layers.Dense(self.dim, name="bottleneck_linear1")
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="bottleneck_norm1")
        self.dropout1 = tf.keras.layers.Dropout(rate=config.dropout_rate, name="bottleneck_dropout1")
        
        self.linear2 = tf.keras.layers.Dense(self.dim, name="bottleneck_linear2")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="bottleneck_norm2")
        self.dropout2 = tf.keras.layers.Dropout(rate=config.dropout_rate, name="bottleneck_dropout2")
        
        # For mean and logvar projection
        self.mean_proj = tf.keras.layers.Dense(self.dim, name="mean_projection")
        self.logvar_proj = tf.keras.layers.Dense(self.dim, name="logvar_projection")
        
        # Handle complex-valued parameters if specified
        if self.use_complex:
            self._init_complex_layers()
    
    
    def _init_complex_layers(self):
        """Initialize complex-valued operations for the bottleneck."""
        # For simplicity, we'll use a proxy approach with real-valued networks
        # but structure the computation to mimic complex operations
        self.real_linear = tf.keras.layers.Dense(self.dim, name="real_linear")
        self.imag_linear = tf.keras.layers.Dense(self.dim, name="imag_linear")
    
    def build(self, input_shape):
        # If input is a tuple of (mean, logvar), get their shape
        if isinstance(input_shape, list) and len(input_shape) == 2:
            shape = input_shape[0]
        else:
            shape = input_shape
            
        # The shape is now known, so build can be called
        self.built = True

    def reparameterize(self, mean, logvar):
        """Implement reparameterization trick for VAE."""
        # Get the dtype of the input tensors
        dtype = mean.dtype
        
        # Use the same dtype for the random noise
        eps = tf.random.normal(shape=tf.shape(mean), dtype=dtype)
        
        # Ensure calculations maintain the same precision
        return mean + tf.cast(tf.exp(0.5 * logvar), dtype) * eps
    
    def complex_reparameterize(self, mean_real, mean_imag, logvar_real, logvar_imag):
        """Complex version of the reparameterization trick."""
        # Get dtype from inputs
        dtype = mean_real.dtype
        
        # Sample real and imaginary parts with matching dtype
        eps_real = tf.random.normal(shape=tf.shape(mean_real), dtype=dtype)
        eps_imag = tf.random.normal(shape=tf.shape(mean_imag), dtype=dtype)
        
        # Compute real and imaginary parts of the sample with consistent types
        real = mean_real + tf.cast(tf.exp(0.5 * logvar_real), dtype) * eps_real
        imag = mean_imag + tf.cast(tf.exp(0.5 * logvar_imag), dtype) * eps_imag
        
        # Return as separate real and imaginary components
        return real, imag
        
    def call(self, inputs, training=False, deterministic=False):
        """Forward pass through the bottleneck.
        
        Args:
            inputs: Tuple of (mean, logvar) or single tensor
            training: Whether in training mode
            deterministic: If True, use mean without sampling
            
        Returns:
            If deterministic=True, returns mean, otherwise returns sampled latent
        """
        # Handle different input types
        if isinstance(inputs, tuple) and len(inputs) == 2:
            mean, logvar = inputs
        else:
            # Single tensor input case - we'll project to mean and logvar
            x = inputs
            x = self.linear1(x)
            x = self.norm1(x)
            x = self.activation_fn(x)
            x = self.dropout1(x, training=training)
            
            x = self.linear2(x)
            x = self.norm2(x)
            x = self.activation_fn(x)
            x = self.dropout2(x, training=training)
            
            # Split into mean and logvar
            mean = self.mean_proj(x)
            logvar = self.logvar_proj(x)
        
        # Apply reparameterization trick if not deterministic
        if not deterministic:
            if self.use_complex:
                # Split mean and logvar into real and imaginary parts
                mean_real, mean_imag = tf.split(mean, 2, axis=-1)
                logvar_real, logvar_imag = tf.split(logvar, 2, axis=-1)
                
                # Apply complex reparameterization
                latent_real, latent_imag = self.complex_reparameterize(
                    mean_real, mean_imag, logvar_real, logvar_imag
                )
                
                # Combine real and imaginary parts
                latent = tf.concat([latent_real, latent_imag], axis=-1)
            else:
                latent = self.reparameterize(mean, logvar)
                
            return latent, mean, logvar
        else:
            # Deterministic path (no sampling)
            return mean, mean, logvar