import tensorflow as tf
import numpy as np

from config import ImageVAEConfig, ImageEncoderDecoderConfig
from encoder import ImageEncoder
from decoder import ImageDecoder
from layers import MultiLayerBottleneck

class ImageVAE(tf.keras.Model):
    """Image Variational Autoencoder model."""
    
    def __init__(self, config: ImageVAEConfig):
        super().__init__()
        self.config = config
        
        # Initialize components
        self._build_encoder(config)
        self._build_decoder(config)
        self._build_bottleneck(config)
        
        # KL weight (for annealing)
        self.kl_weight = tf.Variable(
            config.kl_weight,
            trainable=False,
            dtype=tf.float32,
            name="kl_weight"
        )
        
        # Total training steps counter for curriculum learning
        self.step_counter = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            name="step_counter"
        )
    
    def _build_encoder(self, config):
        """Create image encoder."""
        self.encoder = ImageEncoder(config.image_config)
    
    def _build_decoder(self, config):
        """Create image decoder."""
        self.decoder = ImageDecoder(config.image_config)
    
    def _build_bottleneck(self, config):
        """Create multi-layer bottleneck."""
        self.bottleneck_layers = [
            MultiLayerBottleneck(config.bottleneck_config) 
            for _ in range(config.num_bottleneck_layers)
        ]
    
    def get_config(self):
        """Return configuration of the model for serialization."""
        config = super().get_config()
        # Add custom configuration parameters
        config.update({
            "config": {
                "latent_dim": self.config.latent_dim,
                "num_bottleneck_layers": self.config.num_bottleneck_layers,
                "use_complex_bottleneck": self.config.use_complex_bottleneck,
                "kl_weight": self.config.kl_weight,
                "curriculum_steps": self.config.curriculum_steps,
                "image_config": {
                    "latent_dim": self.config.image_config.latent_dim,
                    "image_size": self.config.image_config.image_size,
                    "channels": self.config.image_config.channels,
                    "use_vit": self.config.image_config.use_vit,
                    "cnn_base_filters": self.config.image_config.cnn_base_filters,
                    "cnn_layers": self.config.image_config.cnn_layers,
                    "dropout_rate": self.config.image_config.dropout_rate
                }
            }
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration dictionary."""
        # Extract the custom config
        model_config = config.get("config", {})
        
        # Recreate the image config
        image_config_dict = model_config.get("image_config", {})
        image_config = ImageEncoderDecoderConfig(**image_config_dict)
        
        # Recreate the main config
        vae_config = ImageVAEConfig(
            latent_dim=model_config.get("latent_dim", 256),
            num_bottleneck_layers=model_config.get("num_bottleneck_layers", 2),
            use_complex_bottleneck=model_config.get("use_complex_bottleneck", False),
            kl_weight=model_config.get("kl_weight", 0.001),
            curriculum_steps=model_config.get("curriculum_steps", 10000),
            image_config=image_config
        )
        
        # Create a new model instance
        model = cls(vae_config)
        
        return model

    def encode(self, inputs, training=False):
        """Encode inputs to latent space.
        
        Args:
            inputs: Input images
            training: Whether in training mode
            
        Returns:
            Mean, logvar, and sampled latent
        """
        # Encode inputs
        mean, logvar = self.encoder(inputs, training=training)
        
        # Apply multi-layer bottleneck
        current_mean, current_logvar = mean, logvar
        for bottleneck_layer in self.bottleneck_layers:
            latent, current_mean, current_logvar = bottleneck_layer(
                (current_mean, current_logvar),
                training=training,
                deterministic=False
            )
        
        return current_mean, current_logvar, latent
    
    def decode(self, latent, training=False):
        """Decode latent vectors to images.
        
        Args:
            latent: Latent vectors
            training: Whether in training mode
            
        Returns:
            Reconstructed images
        """
        return self.decoder(latent, training=training)
    
    def call(self, inputs, training=False):
        """Forward pass through the image VAE.
        
        Args:
            inputs: Input images
            training: Whether in training mode
            
        Returns:
            Reconstructed images, mean, logvar, and latent
        """
        # Encode inputs
        mean, logvar, latent = self.encode(inputs, training=training)
        
        # Decode latent
        outputs = self.decode(latent, training=training)
        
        # Increment step counter during training
        if training:
            self.step_counter.assign_add(1)
            
            # Update KL weight according to curriculum
            if self.config.curriculum_steps > 0:
                progress = tf.cast(self.step_counter, self.kl_weight.dtype) / self.config.curriculum_steps
                progress = tf.minimum(progress, 1.0)
                self.kl_weight.assign(progress * self.config.kl_weight)
        
        return outputs, mean, logvar, latent
    
    def compute_reconstruction_loss(self, inputs, outputs):
        """Compute reconstruction loss.
        
        Args:
            inputs: Input images
            outputs: Reconstructed images
            
        Returns:
            Reconstruction loss
        """
        # MSE or binary cross entropy for images
        # Use binary cross entropy as default for image data in [0, 1] range
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(inputs, outputs)
        )
        
        return loss
    
    def compute_kl_loss(self, mean, logvar):
        """Compute KL divergence loss.
        
        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            KL divergence loss
        """
        kl_loss = -0.5 * tf.reduce_mean(
            1 + logvar - tf.square(mean) - tf.exp(logvar)
        )
        
        return kl_loss
    
    def compute_loss(self, inputs, outputs, mean, logvar):
        """Compute total loss.
        
        Args:
            inputs: Input images
            outputs: Reconstructed images
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Total loss and individual loss components
        """
        # Compute reconstruction loss
        recon_loss = self.compute_reconstruction_loss(inputs, outputs)
        
        # Compute KL divergence loss
        kl_loss = self.compute_kl_loss(mean, logvar)
        
        # Cast kl_loss to match kl_weight's dtype for compatibility with multiplication
        kl_loss_float32 = tf.cast(kl_loss, self.kl_weight.dtype)
        
        # Compute weighted KL loss in float32
        weighted_kl_loss = self.kl_weight * kl_loss_float32
        
        # Cast weighted KL loss back to same dtype as recon_loss for addition
        weighted_kl_loss = tf.cast(weighted_kl_loss, recon_loss.dtype)
        
        # Compute total loss
        total_loss = recon_loss + weighted_kl_loss
        
        return total_loss, {
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "total": total_loss
        }
    
    def train_step(self, data):
        """Custom training step.
        
        Args:
            data: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of loss metrics
        """
        # Unpack data
        inputs, _ = data  # For VAE, targets = inputs
        
        with tf.GradientTape() as tape:
            # Forward pass
            outputs, mean, logvar, latent = self(inputs, training=True)
            
            # Compute loss
            total_loss, loss_dict = self.compute_loss(inputs, outputs, mean, logvar)
            
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        metrics_dict = {k: v for k, v in loss_dict.items()}
        metrics_dict["kl_weight"] = self.kl_weight
        
        return metrics_dict
    
    def test_step(self, data):
        """Custom test/validation step.
        
        Args:
            data: Tuple of (inputs, targets)
            
        Returns:
            Dictionary of loss metrics
        """
        # Unpack data
        inputs, _ = data  # For VAE, targets = inputs
        
        # Forward pass (no training)
        outputs, mean, logvar, latent = self(inputs, training=False)
        
        # Compute loss
        total_loss, loss_dict = self.compute_loss(inputs, outputs, mean, logvar)
        
        # Return metrics
        return loss_dict
    
    def generate(self, latent):
        """Generate images from latent vectors.
        
        Args:
            latent: Latent vectors
            
        Returns:
            Generated images
        """
        return self.decode(latent, training=False)
    
    def interpolate(self, img_a, img_b, steps=10):
        """Interpolate between two images in latent space.
        
        Args:
            img_a: First image
            img_b: Second image
            steps: Number of interpolation steps
            
        Returns:
            List of interpolated images
        """
        # Encode images to latent space
        mean_a, logvar_a, latent_a = self.encode(tf.expand_dims(img_a, 0))
        mean_b, logvar_b, latent_b = self.encode(tf.expand_dims(img_b, 0))
        
        # Get the dtype of the latent vectors to ensure type consistency
        dtype = latent_a.dtype
        
        # Create interpolation weights with matching dtype
        alphas = tf.cast(tf.linspace(0.0, 1.0, steps), dtype)
        
        # Interpolate and generate
        outputs = []
        
        for alpha in alphas:
            # Linear interpolation with matching types
            latent_interp = (1 - alpha) * latent_a + alpha * latent_b
            
            # Generate outputs
            output = self.generate(latent_interp)[0]
            outputs.append(output)
                
        return outputs
    
    def sample(self, n_samples=1):
        """Sample random images from the latent space.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated images
        """
        # Sample latent vectors from a standard normal distribution
        latent = tf.random.normal([n_samples, self.config.latent_dim])
        
        # Generate images
        return self.generate(latent)
    
    def save_model(self, filepath):
        """Save model to disk using TensorFlow SavedModel format."""
        # Save the model weights
        self.save_weights(filepath + ".weights.h5")
        
        # Save the model config
        import json
        with open(filepath + "_config.json", "w") as f:
            json.dump(self.get_config(), f)

    @classmethod
    def load_model(cls, filepath):
        """Load model from disk."""
        # Load the config
        import json
        with open(filepath + "_config.json", "r") as f:
            config = json.load(f)
        
        # Create a new model from config
        model = cls.from_config(config)
        
        # Load the weights
        model.load_weights(filepath + ".weights.h5")
        
        return model
    
    def save_model_weights(self, filepath):
        """Save model weights to disk."""
        self.save_weights(filepath)
    
    @classmethod
    def load_model_weights(cls, config, filepath):
        """Load model from weights.
        
        Args:
            config: An ImageVAEConfig instance
            filepath: Path to the saved weights
        """
        model = cls(config)
        
        # Initialize with a dummy input
        dummy_input = tf.zeros((1, *config.image_config.image_size, config.image_config.channels))
        _ = model(dummy_input)
        
        # Load weights
        model.load_weights(filepath)
        
        return model
