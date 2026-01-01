import os
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import config
from .dataset import DenseFusionDataset
from .models import DenseFusionNetwork
from .loss import DenseFusionLoss
from .utils import load_yaml_file, load_depth_scale_factor


def create_data_loaders():
    dataset_config = load_yaml_file(config.DATA_YAML_PATH)
    train_dataset = DenseFusionDataset(dataset_config, split='train')
    val_dataset = DenseFusionDataset(dataset_config, split='val')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    return train_loader, val_loader, train_dataset


def save_model_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, is_best=False):
    os.makedirs(config.MODELS_SAVE_DIR, exist_ok=True)
    checkpoint_data = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': getattr(scheduler, 'state_dict', lambda: None)(), 'train_loss': train_loss, 'val_loss': val_loss, 'config': {'num_points': config.NUM_POINTS, 'patch_size': config.PATCH_SIZE, 'batch_size': config.BATCH_SIZE, 'learning_rate': config.LEARNING_RATE}}
    if is_best:
        best_path = os.path.join(config.MODELS_SAVE_DIR, f'{config.MODELS_SAVE_DIR}_best.pth')
        torch.save(checkpoint_data, best_path)
        return best_path
    else:
        epoch_path = os.path.join(config.CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch:03d}.pth')
        os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
        torch.save(checkpoint_data, epoch_path)
        return epoch_path


def train_densefusion():
    config.setup_environment()
    train_loader, val_loader, train_dataset = create_data_loaders()
    dataset_config = load_yaml_file(config.DATA_YAML_PATH)
    num_classes = len(dataset_config.get('names', []))
    model = DenseFusionNetwork(num_objects=num_classes, use_transformer=config.USE_TRANSFORMER_FUSION).to(config.DEVICE)
    loss_fn = DenseFusionLoss(train_dataset.object_models, config.SYMMETRIC_LIST)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION and torch.cuda.is_available() else None
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            rgb = batch['rgb'].to(config.DEVICE)
            points = batch['points'].to(config.DEVICE)
            gt_poses = batch['gt_pose'].to(config.DEVICE)
            class_ids = batch['class_id'].to(config.DEVICE)
            optimizer.zero_grad()
            if scaler is not None:
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
            pbar.set_postfix({'Loss': f"{loss_dict['total_loss']:.6f}", 'ADD': f"{loss_dict['add_loss']:.6f}"})
        avg_train_loss = train_loss_accum / max(train_batches, 1)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(config.DEVICE)
                points = batch['points'].to(config.DEVICE)
                gt_poses = batch['gt_pose'].to(config.DEVICE)
                class_ids = batch['class_id'].to(config.DEVICE)
                pred_poses, pred_confs = model(rgb, points)
                total_loss, loss_dict = loss_fn(pred_poses, gt_poses, pred_confs, class_ids)
                val_loss_accum += loss_dict['total_loss']
                val_batches += 1
        avg_val_loss = val_loss_accum / max(val_batches, 1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        save_model_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss, is_best)
    return {'model': model, 'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}
