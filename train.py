# FILE: train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

# Local imports
from models.densefusion import DenseFusionNetwork
from dataset.dataset import DenseFusionDataset
from utils.loss import DenseFusionLoss
from utils.utils import (
    load_dataset_config,
    save_model_checkpoint,
    load_checkpoint,
    cleanup_memory,
    Config,
    create_directories,
)

def create_data_loaders(config):
    """Create optimized data loaders"""
    dataset_config = load_dataset_config(config.LINEMOD_ROOT)

    train_dataset = DenseFusionDataset(
        dataset_config, split='train', config=config, use_augmentation=False,
        use_segmentation=False  # Disabled for training stability
    )

    val_dataset = DenseFusionDataset(
        dataset_config, split='val', config=config, use_augmentation=False,
        use_segmentation=False
    )

    # Use num_workers=0 for Colab/local compatibility without spawn issues
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False
    )

    return train_loader, val_loader, train_dataset

def create_training_plots(train_losses, val_losses, save_path=None):
    """Create and save training progress plots"""
    try:
        plt.figure(figsize=(12, 4))
        epochs = range(1, len(train_losses) + 1)

        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Log scale
        plt.subplot(1, 2, 2)
        plt.semilogy(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.semilogy(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.title('Training Progress (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"Failed to create plots: {e}")

def train_densefusion(config):
    """Complete DenseFusion training pipeline"""
    print("=" * 60)
    print("DENSEFUSION TRAINING PIPELINE")
    print("=" * 60)

    # Setup
    config.setup_environment()
    start_time = datetime.datetime.now()
    create_directories(config)

    # Load data
    train_loader, val_loader, train_dataset = create_data_loaders(config)
    dataset_config = load_dataset_config(config.LINEMOD_ROOT)
    num_classes = len(dataset_config.get('names', []))

    print(f"Training setup:")
    print(f"  Classes: {num_classes}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Device: {config.DEVICE}")

    # Initialize model and training components
    model = DenseFusionNetwork(
        num_objects=num_classes,
        use_transformer=config.USE_TRANSFORMER_FUSION,
        config=config
    ).to(config.DEVICE)
    loss_fn = DenseFusionLoss(train_dataset.object_models, config=config)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    # Mixed precision scaler
    scaler = None
    if config.USE_MIXED_PRECISION and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    # Training state
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\nStarting training...")

    try:
        for epoch in range(config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
            print("-" * 40)

            # Training phase
            model.train()
            train_loss_accum = 0.0
            train_batches = 0
            loss_fn.reset_stats()
            train_pbar = tqdm(train_loader, desc=f"Training", leave=False)

            for batch_idx, batch in enumerate(train_pbar):
                try:
                    rgb = batch['rgb'].to(config.DEVICE, non_blocking=True)
                    points = batch['points'].to(config.DEVICE, non_blocking=True)
                    gt_poses = batch['gt_pose'].to(config.DEVICE, non_blocking=True)
                    class_ids = batch['class_id'].to(config.DEVICE, non_blocking=True)
                    optimizer.zero_grad()

                    if scaler:
                        with torch.cuda.amp.autocast():
                            pred_poses, pred_confs = model(rgb, points)
                            total_loss, loss_dict = loss_fn(pred_poses, gt_poses, pred_confs, class_ids)
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        pred_poses, pred_confs = model(rgb, points)
                        total_loss, loss_dict = loss_fn(pred_poses, gt_poses, pred_confs, class_ids)
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    train_loss_accum += loss_dict['total_loss']
                    train_batches += 1
                    train_pbar.set_postfix({'Loss': f"{loss_dict['total_loss']:.6f}", 'ADD': f"{loss_dict['add_loss']:.6f}"})

                    if batch_idx % 50 == 0:
                        cleanup_memory()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM at batch {batch_idx}, cleaning up...")
                        cleanup_memory()
                        optimizer.zero_grad()
                    else:
                        print(f"\nError in training batch {batch_idx}: {e}")

            avg_train_loss = train_loss_accum / max(train_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss_accum = 0.0
            val_batches = 0
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            with torch.no_grad():
                for batch in val_pbar:
                    rgb = batch['rgb'].to(config.DEVICE, non_blocking=True)
                    points = batch['points'].to(config.DEVICE, non_blocking=True)
                    gt_poses = batch['gt_pose'].to(config.DEVICE, non_blocking=True)
                    class_ids = batch['class_id'].to(config.DEVICE, non_blocking=True)

                    if scaler:
                        with torch.cuda.amp.autocast():
                            pred_poses, pred_confs = model(rgb, points)
                            _, loss_dict = loss_fn(pred_poses, gt_poses, pred_confs, class_ids)
                    else:
                        pred_poses, pred_confs = model(rgb, points)
                        _, loss_dict = loss_fn(pred_poses, gt_poses, pred_confs, class_ids)
                    
                    val_loss_accum += loss_dict['total_loss']
                    val_batches += 1
                    val_pbar.set_postfix({'Val Loss': f"{loss_dict['total_loss']:.4f}"})

            avg_val_loss = val_loss_accum / max(val_batches, 1)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1} Results: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"  ✓ NEW BEST MODEL!")
            
            save_model_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss, is_best, config)

            if (epoch + 1) % 2 == 0:
                plot_path = os.path.join(config.MODELS_SAVE_DIR, 'training_progress.png')
                create_training_plots(train_losses, val_losses, plot_path)

            cleanup_memory()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f"\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Models saved to: {config.MODELS_SAVE_DIR}")
        final_plot_path = os.path.join(config.MODELS_SAVE_DIR, f'final_training_plot.png')
        create_training_plots(train_losses, val_losses, final_plot_path)


if __name__ == '__main__':
    # Initialize configuration
    config = Config()
    
    # Verify paths before starting
    if not config.verify_paths():
        print("\n⚠ Please update the paths in the utils/utils.py file before proceeding")
    else:
        train_densefusion(config)