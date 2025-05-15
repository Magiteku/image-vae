from typing import Dict, List, Optional, Tuple, Union, Any

class Config:
    """Base configuration class for model components."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ImageVAEConfig(Config):
    """Configuration for the Image VAE model."""
    def __init__(
        self,
        latent_dim: int = 512,
        num_bottleneck_layers: int = 4,
        use_complex_bottleneck: bool = False,
        kl_weight: float = 0.001,
        curriculum_steps: int = 10000,
        **kwargs
    ):
        self.latent_dim = latent_dim
        self.num_bottleneck_layers = num_bottleneck_layers
        self.use_complex_bottleneck = use_complex_bottleneck
        self.kl_weight = kl_weight
        self.curriculum_steps = curriculum_steps
        
        # Create image encoder/decoder config with defaults
        self.image_config = ImageEncoderDecoderConfig(latent_dim=latent_dim)
        
        # Bottleneck config
        self.bottleneck_config = BottleneckConfig(
            latent_dim=latent_dim,
            use_complex=use_complex_bottleneck
        )
        
        super().__init__(**kwargs)

class BottleneckConfig(Config):
    """Configuration for the bottleneck layers."""
    def __init__(
        self,
        latent_dim: int = 512,
        use_complex: bool = False,
        activation: str = "leaky_relu",
        leaky_alpha: float = 0.2,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        self.latent_dim = latent_dim
        self.use_complex = use_complex
        self.activation = activation
        self.leaky_alpha = leaky_alpha
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

class ImageEncoderDecoderConfig(Config):
    """Configuration for Image encoder and decoder."""
    def __init__(
        self,
        latent_dim: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        use_vit: bool = True,
        patch_size: int = 16,
        vit_layers: int = 6,
        vit_heads: int = 8,
        vit_ff_dim: int = 1024,
        cnn_base_filters: int = 64,
        cnn_layers: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.use_vit = use_vit
        self.patch_size = patch_size
        self.vit_layers = vit_layers
        self.vit_heads = vit_heads
        self.vit_ff_dim = vit_ff_dim
        self.cnn_base_filters = cnn_base_filters
        self.cnn_layers = cnn_layers
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)
