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
    parser = argparse.ArgumentParser(description="Train an Image VAE on FFHQ or other datasets")
    parser.add_argument("--dataset", type=str, default="ffhq", 
                       help="Dataset to use (ffhq, cifar100, cifar10, etc.)")
    parser.add_argument("--latent-dim", type=int, default=256, 
                       help="Dimension of the latent space")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=128, 
                       help="Image size (square)")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum number of samples to load")
    parser.add_argument("--kl-weight", type=float, default=0.001, 
                       help="Weight for the KL divergence loss")
    parser.add_argument("--output-dir", type=str, default="output", 
                       help="Directory to save outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", 
                       help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", 
                       help="Directory for TensorBoard logs")
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate visualizations after training")
    parser.add_argument("--use-mixed-precision", action="store_true",
                       help="Enable mixed precision training")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Configure mixed precision if requested
    if args.use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')
        print("Using float32 precision")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create configuration optimized for the chosen dataset
    if args.dataset.lower() == "ffhq":
        # FFHQ-optimized configuration
        print("Configuring for FFHQ face dataset...")
        image_config = ImageEncoderDecoderConfig(
            latent_dim=args.latent_dim,
            image_size=(args.img_size, args.img_size),
            channels=3,
            use_vit=False,  # CNN works well for faces
            cnn_base_filters=64,
            cnn_layers=4,
            dropout_rate=0.1
        )
        
        config = ImageVAEConfig(
            latent_dim=args.latent_dim,
            num_bottleneck_layers=2,  # More layers for complex face features
            use_complex_bottleneck=False,
            kl_weight=args.kl_weight,
            curriculum_steps=10000,  # Curriculum learning for better face generation
            image_config=image_config
        )
    else:
        # Default configuration for other datasets
        print(f"Configuring for {args.dataset} dataset...")
        image_config = ImageEncoderDecoderConfig(
            latent_dim=args.latent_dim,
            image_size=(args.img_size, args.img_size),
            channels=3,
            use_vit=False,
            cnn_base_filters=64,
            cnn_layers=4,
            dropout_rate=0.1
        )
        
        config = ImageVAEConfig(
            latent_dim=args.latent_dim,
            num_bottleneck_layers=2,
            use_complex_bottleneck=False,
            kl_weight=args.kl_weight,
            curriculum_steps=5000,
            image_config=image_config
        )
    
    # Create model
    print("Creating VAE model...")
    model = create_image_vae(config)
    model.summary()
    
    # Create data generators
    print(f"Loading {args.dataset} dataset...")
    train_data, val_data = create_data_generators(
        config.image_config, 
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        dataset_type=args.dataset
    )
    
    # Display sample images
    sample_batch = next(iter(train_data))[0][:5]
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(sample_batch[i])
        plt.axis('off')
        plt.title(f'Sample {i+1}')
    plt.suptitle(f'Sample {args.dataset.upper()} Training Images')
    plt.savefig(os.path.join(args.output_dir, f"{args.dataset}_samples.png"))
    plt.show()
    
    # Train model
    print("Starting training...")
    history = train_image_vae(
        model,
        train_data,
        val_data,
        config,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    model_save_path = os.path.join(args.checkpoint_dir, f"{args.dataset}_vae_final")
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history["total"], label="Train")
    plt.plot(history.history["val_total"], label="Validation")
    plt.title(f"Total Loss ({args.dataset.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history["reconstruction"], label="Train")
    plt.plot(history.history["val_reconstruction"], label="Validation")
    plt.title(f"Reconstruction Loss ({args.dataset.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history["kl"], label="Train")
    plt.plot(history.history["val_kl"], label="Validation")
    plt.title(f"KL Divergence Loss ({args.dataset.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.dataset}_training_history.png"))
    plt.show()
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        
        # Get some validation samples
        val_batch = next(iter(val_data))[0][:10]
        
        # Visualize reconstructions
        print("Creating reconstruction visualizations...")
        recon_fig = visualize_reconstructions(model, val_batch)
        recon_fig.savefig(os.path.join(args.output_dir, f"{args.dataset}_reconstructions.png"))
        plt.show()
        
        # Visualize random samples
        print("Creating random sample visualizations...")
        n_samples = 16 if args.dataset.lower() == "ffhq" else 10
        grid_size = (4, 4) if args.dataset.lower() == "ffhq" else (2, 5)
        samples_fig = visualize_samples(model, n_samples=n_samples, grid_size=grid_size)
        samples_fig.savefig(os.path.join(args.output_dir, f"{args.dataset}_random_samples.png"))
        plt.show()
        
        # Visualize latent space
        print("Creating latent space visualizations...")
        val_images = np.concatenate([next(iter(val_data))[0] for _ in range(10)])[:1000]
        latent_fig = visualize_latent_space(model, val_images)
        latent_fig.savefig(os.path.join(args.output_dir, f"{args.dataset}_latent_space.png"))
        plt.show()
        
        # Visualize interpolation
        print("Creating interpolation visualizations...")
        steps = 10 if args.dataset.lower() == "ffhq" else 8
        interp_fig = visualize_interpolation(model, val_batch[0], val_batch[1], steps=steps)
        interp_fig.savefig(os.path.join(args.output_dir, f"{args.dataset}_interpolation.png"))
        plt.show()
        
        # Special visualization for FFHQ (face-specific)
        if args.dataset.lower() == "ffhq":
            print("Creating FFHQ-specific face generation visualizations...")
            
            # Generate high-quality face samples
            face_samples = model.sample(n_samples=25)
            face_samples_np = tf.cast(face_samples, tf.float32).numpy()
            
            fig, axes = plt.subplots(5, 5, figsize=(15, 15))
            for i in range(5):
                for j in range(5):
                    idx = i * 5 + j
                    axes[i, j].imshow(face_samples_np[idx])
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'Face {idx+1}', fontsize=8)
            
            plt.suptitle('Generated Faces from FFHQ VAE', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "ffhq_generated_faces_grid.png"), dpi=150)
            plt.show()
    
    print(f"Training complete! Results saved to {args.output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Training samples: {len(train_data) * args.batch_size}")
    print(f"Validation samples: {len(val_data) * args.batch_size}")
    print(f"Epochs trained: {args.epochs}")
    print(f"Final reconstruction loss: {history.history['reconstruction'][-1]:.4f}")
    print(f"Final KL loss: {history.history['kl'][-1]:.4f}")
    print(f"Model saved to: {model_save_path}")
    print("="*50)

if __name__ == "__main__":
    main()