import tensorflow as tf
import numpy as np

from config import ImageVAEConfig
from model import ImageVAE
from callbacks import CurriculumLearningCallback

def create_image_vae(config=None):
    """Create an Image VAE model with default or custom configuration.
    
    Args:
        config: Optional custom configuration (uses default if None)
        
    Returns:
        Initialized ImageVAE model
    """
    if config is None:
        config = ImageVAEConfig()
        
    model = ImageVAE(config)
    
    # Initialize with a dummy batch to build the model
    dummy_input = tf.zeros((1, *config.image_config.image_size, config.image_config.channels), 
                           dtype=tf.float32)
    _ = model(dummy_input)
    
    return model

def prepare_callbacks(model, config):
    """Prepare training callbacks for the Image VAE.
    
    Args:
        model: ImageVAE instance
        config: Model configuration
        
    Returns:
        List of callbacks for training
    """
    callbacks = []
    
    # Curriculum learning
    if config.curriculum_steps > 0:
        callbacks.append(
            CurriculumLearningCallback(
                total_steps=config.curriculum_steps,
                final_kl_weight=config.kl_weight
            )
        )
    
    # TensorBoard logging
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/image_vae",
            update_freq="batch",
            histogram_freq=1
        )
    )
    
    # Model checkpointing
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/image_vae/model_{epoch:02d}.keras",
            save_weights_only=False,
            save_best_only=True,
            monitor="val_total",
            mode="min"
        )
    )
    
    # Early stopping
    callbacks.append(
    tf.keras.callbacks.EarlyStopping(
        monitor="val_total",
        patience=10,
        restore_best_weights=True,
        mode='min'  # Add this parameter to specify we want to minimize the monitored metric
    )
)
    
    return callbacks

def train_image_vae(model, train_data, val_data, config, epochs=100, batch_size=32):
    """Train the Image VAE model.
    
    Args:
        model: ImageVAE instance
        train_data: Training data generator
        val_data: Validation data generator
        config: Model configuration
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Training history
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        run_eagerly=False  # Set to True for debugging
    )
    
    # Prepare callbacks
    callbacks = prepare_callbacks(model, config)
    
    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history

def visualize_reconstructions(model, images, n_samples=10):
    """Visualize original images and their reconstructions."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Limit the number of samples
    images = images[:n_samples]
    
    # Get reconstructions
    reconstructions, _, _, _ = model(images, training=False)
    
    # Convert tensors to numpy arrays with supported dtype
    images_np = tf.cast(images, tf.float32).numpy()
    reconstructions_np = tf.cast(reconstructions, tf.float32).numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    
    # Plot original images
    for i in range(n_samples):
        axes[0, i].imshow(images_np[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
    
    # Plot reconstructions
    for i in range(n_samples):
        axes[1, i].imshow(reconstructions_np[i])
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    return fig

def visualize_samples(model, n_samples=10, grid_size=None):
    """Visualize random samples from the latent space."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate samples
    samples = model.sample(n_samples)
    
    # Convert to numpy with supported dtype
    samples_np = tf.cast(samples, tf.float32).numpy()
    
    # Determine grid size
    if grid_size is None:
        grid_size = (1, n_samples)
    rows, cols = grid_size
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # Plot samples
    sample_idx = 0
    for i in range(rows):
        for j in range(cols):
            if sample_idx < n_samples:
                if rows == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                ax.imshow(samples_np[sample_idx])
                ax.axis("off")
                sample_idx += 1
    
    plt.tight_layout()
    return fig

def visualize_latent_space(model, images, labels=None, n_samples=1000):
    """Visualize the latent space using PCA or t-SNE.
    
    Args:
        model: Trained ImageVAE model
        images: Array of input images
        labels: Optional array of labels for coloring points
        n_samples: Maximum number of samples to visualize
        
    Returns:
        Matplotlib figure with PCA and t-SNE visualizations
    """
    # Import visualization libraries
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Limit the number of samples
    images = images[:n_samples]
    if labels is not None:
        labels = labels[:n_samples]
    
    # Encode images to latent space
    mean, _, _ = model.encode(images)
    latent_vectors = mean.numpy()
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latent_vectors)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(latent_vectors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PCA results
    if labels is not None:
        scatter_pca = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter_pca, ax=axes[0])
    else:
        axes[0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    axes[0].set_title("PCA of Latent Space")
    
    # Plot t-SNE results
    if labels is not None:
        scatter_tsne = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter_tsne, ax=axes[1])
    else:
        axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
    axes[1].set_title("t-SNE of Latent Space")
    
    plt.tight_layout()
    return fig

def visualize_interpolation(model, img_a, img_b, steps=10):
    """Visualize interpolation between two images in latent space."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get interpolated images
    interpolated_images = model.interpolate(img_a, img_b, steps)
    
    # Convert to numpy arrays with supported dtype
    interpolated_np = [tf.cast(img, tf.float32).numpy() for img in interpolated_images]
    
    # Create figure
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    
    # Plot interpolated images
    for i in range(steps):
        axes[i].imshow(interpolated_np[i])
        axes[i].axis("off")
        if i == 0:
            axes[i].set_title("Start")
        elif i == steps - 1:
            axes[i].set_title("End")
    
    plt.tight_layout()
    return fig
