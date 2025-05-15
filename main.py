import tensorflow as tf
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from config import ImageVAEConfig, ImageEncoderDecoderConfig
from model import ImageVAE
from data_generator import create_data_generators
from training_utils import (
    create_image_vae,
    train_image_vae,
    visualize_reconstructions,
    visualize_samples,
    visualize_latent_space,
    visualize_interpolation
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an Image VAE")
    parser.add_argument("--latent-dim", type=int, default=256, help="Dimension of the latent space")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=64, help="Image size (square)")
    parser.add_argument("--kl-weight", type=float, default=0.001, help="Weight for the KL divergence loss")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for TensorBoard logs")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after training")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create configuration
    image_config = ImageEncoderDecoderConfig(
        latent_dim=args.latent_dim,
        image_size=(args.img_size, args.img_size),
        channels=3,
        use_vit=False,  # Use CNN for simplicity
        cnn_base_filters=64,
        cnn_layers=4,
        dropout_rate=0.1
    )
    
    config = ImageVAEConfig(
        latent_dim=args.latent_dim,
        num_bottleneck_layers=2,
        use_complex_bottleneck=False,
        kl_weight=args.kl_weight,
        curriculum_steps=10000,
        image_config=image_config
    )
    
    # Create model
    model = create_image_vae(config)
    model.summary()
    
    # Create data generators
    train_data, val_data = create_data_generators(config.image_config, batch_size=args.batch_size)
    
    # Train model
    history = train_image_vae(
        model,
        train_data,
        val_data,
        config,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_save_path = os.path.join(args.checkpoint_dir, "image_vae_final")
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history["reconstruction"], label="Train")
    plt.plot(history.history["val_reconstruction"], label="Validation")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history["kl"], label="Train")
    plt.plot(history.history["val_kl"], label="Validation")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_history.png"))
    
    # Generate visualizations if requested
    if args.visualize:
        # Get some validation samples
        val_batch = next(iter(val_data))[0][:10]  # Get 10 validation images
        
        # Visualize reconstructions
        recon_fig = visualize_reconstructions(model, val_batch)
        recon_fig.savefig(os.path.join(args.output_dir, "reconstructions.png"))
        
        # Visualize random samples
        samples_fig = visualize_samples(model, n_samples=10)
        samples_fig.savefig(os.path.join(args.output_dir, "random_samples.png"))
        
        # Visualize latent space (needs more samples)
        val_images = np.concatenate([next(iter(val_data))[0] for _ in range(10)])[:1000]
        latent_fig = visualize_latent_space(model, val_images)
        latent_fig.savefig(os.path.join(args.output_dir, "latent_space.png"))
        
        # Visualize interpolation between two images
        interp_fig = visualize_interpolation(model, val_batch[0], val_batch[1])
        interp_fig.savefig(os.path.join(args.output_dir, "interpolation.png"))
    
    print("Done!")

if __name__ == "__main__":
    main()
