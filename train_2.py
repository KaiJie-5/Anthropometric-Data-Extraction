# Ignore Stage 2 training code, still in experiment phase, not part of training the final submission models.

import sys
sys.path.append("/iridisfs/scratch/ejl1e22/Anthropometric-Data-Extraction/src_1/")

import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import shutil
import json
import random
import argparse
from pathlib import Path
from torchvision import transforms

# Import your anthropometric modules
from Trainer_1 import create_stage1_trainer, create_stage2_trainer
from extractor import SVCNN_RGBD, AnthropometricViewGCN, SVCNNEnsemble
from wrapper import (
    create_train_val_test_split, 
    create_stage1_datasets, 
    create_stage2_single_config_dataset,
    get_standardization_info,
    standardize_metrics
)
        
class ApplyToRGB:
    """
    Applies a given torchvision transform to the RGB channels of a 4-channel tensor.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, tensor_4d):
        # Separate the RGB and Depth channels
        rgb_channels = tensor_4d[:3, :, :]
        depth_channel = tensor_4d[3:, :, :]
        
        # Apply the transform to the RGB channels
        transformed_rgb = self.transform(rgb_channels)
        
        # Recombine the transformed RGB with the original depth channel
        return torch.cat([transformed_rgb, depth_channel], dim=0)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noise_level = random.uniform(0.0, 0.1)
        return tensor + (noise * noise_level)
        
def seed_torch(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

def create_folder(log_dir):
    """Create log directory if it doesn't exist, otherwise use existing one."""
    if os.path.exists(log_dir):
        print(f'Log directory {log_dir} already exists. Using existing directory.')
    else:
        os.makedirs(log_dir, exist_ok=True)
        print(f'Created new log directory: {log_dir}')

def setup_multi_gpu(model, device):
    """Setup multi-GPU training if available."""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    else:
        print(f"Using single GPU: {device}")
    
    model.to(device)
    return model

def save_config(args, log_dir):
    """Save training configuration to JSON file."""
    config_path = os.path.join(log_dir, 'config.json')
    config_dict = vars(args).copy()
    
    # Add system info
    config_dict['num_gpus'] = torch.cuda.device_count()
    config_dict['cuda_available'] = torch.cuda.is_available()
    
    # Add dataset info
    std_info = get_standardization_info()
    config_dict['anthropometrics_mean'] = std_info['mean'].tolist()
    config_dict['anthropometrics_std'] = std_info['std'].tolist()
    config_dict['num_features'] = std_info['num_features']
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")

def print_dataset_info(train_subjects, val_subjects, train_loader_s1, val_loader_s1, 
                      train_loader_s2, val_loader_s2):
    """Print comprehensive dataset information."""
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Total subjects: {len(train_subjects) + len(val_subjects)}")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Train/Val split ratio: {len(train_subjects)/(len(train_subjects)+len(val_subjects)):.2f}")
    
    print("\nStage 1 (Single-view):")
    print(f"  Training samples: {len(train_loader_s1.dataset):,}")
    print(f"  Validation samples: {len(val_loader_s1.dataset):,}")
    print(f"  Samples per subject: ~{len(train_loader_s1.dataset)//len(train_subjects)}")
    
    print("\nStage 2 (Multi-view):")
    print(f"  Training samples: {len(train_loader_s2.dataset):,}")
    print(f"  Validation samples: {len(val_loader_s2.dataset):,}")
    print(f"  Configurations per subject: 3 (3-view, 36-view, 72-view)")
    print("=" * 80)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train ViewGCN for Anthropometric Measurement Prediction")
    
    # Data paths
    parser.add_argument("--images_root", type=str, required=True,
                       help="Path to root directory containing subject image folders")
    parser.add_argument("--anthropometrics_path", type=str, required=True,
                       help="Path to anthropometrics CSV file")
    parser.add_argument("--front_images_path", type=str, required=True,
                       help="Path to front images CSV file")
    
    # Model configuration
    parser.add_argument("--cnn_name", type=str, default="resnet18", 
                       choices=["resnet18", "resnet34", "resnet50"],
                       help="CNN backbone architecture")
    parser.add_argument("--no_pretraining", action='store_true',
                       help="Disable ImageNet pretraining")
    
    # Training configuration
    parser.add_argument("--batch_size_s1", type=int, default=32,
                       help="Batch size for Stage 1 (single-view) training")
    parser.add_argument("--batch_size_s2", type=int, default=4,
                       help="Batch size for Stage 2 (multi-view) training")
    parser.add_argument("--epochs_s1", type=int, default=30,
                       help="Number of epochs for Stage 1 training")
    parser.add_argument("--epochs_s2", type=int, default=30,
                       help="Number of epochs for Stage 2 training")
    parser.add_argument("--lr_s1", type=float, default=0.0001,
                       help="Learning rate for Stage 1")
    parser.add_argument("--lr_s2", type=float, default=0.0005,
                       help="Learning rate for Stage 2")
    parser.add_argument("--weight_decay", type=float, default=8e-4,
                       help="Weight decay for optimization")
    
    # Data split configuration
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Ratio of subjects used for training")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Ratio of subjects used for validation. The remainder will be used for the test set.")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
                       
    # Experiment configuration
    parser.add_argument("--name", type=str, default="anthropometric_viewgcn",
                       help="Experiment name (used for log directories)")
    parser.add_argument("--resume_stage1", type=str, default=None,
                       help="Path to Stage 1 model to resume from")
    parser.add_argument("--resume_stage2", type=str, default=None,
                       help="Path to Stage 2 model to resume from")
    parser.add_argument("--skip_stage1", action='store_true',
                       help="Skip Stage 1 training (requires resume_stage1)")
    
    # Stage 2 config name
    parser.add_argument("--config_name", type=str, default="frontal",
                       choices=["frontal", "full", "minimal"],
                       help="Configuration name for Stage 2 dataset")
    
    # System configuration
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loading workers")
                       
    parser.add_argument("--task_type", type=str, default="anthropometric",
                       choices=["anthropometric", "landmark"],
                       help="Task type: 'anthropometric' or 'landmark'")
                       
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["left_ear", "right_ear", "head"],
                        help="Specify which model to train.")
                        
    # Add AFTER --front_images_path argument:
    parser.add_argument("--landmarks_path", type=str, default=None,
                       help="Path to landmarks directory (required for landmark task)")
    
    args = parser.parse_args()
    
    if args.model_type in ['left_ear', 'right_ear']:
        nfeatures = 5
    elif args.model_type == 'head':
        nfeatures = 1
        
    log_dir = f"{args.name}_{args.model_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup device and random seed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    seed_torch(args.random_seed)
    
    # Validate arguments
    if args.skip_stage1 and args.resume_stage1 is None:
        raise ValueError("--skip_stage1 requires --resume_stage1 to be specified")
    
    # NEW: Task-specific validation
    if args.task_type == 'landmark' and args.landmarks_path is None:
        raise ValueError("--landmarks_path is required for landmark task")
    
    # NEW: Set output dimension based on task
    output_dim = 22 if args.task_type == 'landmark' else 11
    print(f"\nTask Configuration:")
    print(f"  Task type: {args.task_type}")
    print(f"  Output dimension: {output_dim}")
    
    # Create main experiment directory
    main_log_dir = args.name
    create_folder(main_log_dir)
    save_config(args, main_log_dir)
    
    # Create train/validation/test split
    print("Creating train/validation/test split...")
    train_subjects, val_subjects, _ = create_train_val_test_split(
        images_root=args.images_root,
        anthropometrics_path=args.anthropometrics_path,
        front_images_path=args.front_images_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )

    train_image_transform = transforms.Compose([
        # PIL-based transforms
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),

        # Convert to Tensor
        transforms.ToTensor(),

        # Normalization must be last
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
        ])

    target_transform = lambda metrics: standardize_metrics(metrics, args.model_type)
    
    # Create Stage 1 datasets by specifying the model_type
    print(f"Creating Stage 1 datasets for model_type: '{args.model_type}'...")
    train_dataset_s1, val_dataset_s1 = create_stage1_datasets(
        images_root=args.images_root,
        anthropometrics_path=args.anthropometrics_path,
        front_images_path=args.front_images_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        model_type=args.model_type,
        target_transform=target_transform,
        train_transform=train_image_transform,
        val_transform=val_image_transform 
    )
    
#    # Create Stage 2 datasets (multi-view, all configurations)
#    print("Creating Stage 2 datasets...")
#    train_dataset_s2, val_dataset_s2 = create_stage2_datasets(
#        images_root=args.images_root,
#        anthropometrics_path=args.anthropometrics_path,
#        front_images_path=args.front_images_path,
#        train_subjects=train_subjects,
#        val_subjects=val_subjects
#    )
    
    print(f"\nCreating Stage 2 dataset for {args.config_name} configuration...")
    
    # Create Stage 2 datasets (single configuration)
    train_dataset_s2, val_dataset_s2 = create_stage2_single_config_dataset(
        images_root=args.images_root,
        anthropometrics_path=args.anthropometrics_path,
        front_images_path=args.front_images_path,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        config_name=args.config_name,
        train_transform=train_image_transform,
        val_transform=val_image_transform,
        train_target_transform=standardize_metrics,
        val_target_transform=standardize_metrics
    )

    # Create data loaders
    train_loader_s1 = torch.utils.data.DataLoader(
        train_dataset_s1, 
        batch_size=args.batch_size_s1, 
        shuffle=True, 
        num_workers=args.num_workers,
        #pin_memory=True if device == 'cuda' else False
        pin_memory = False
    )
    
    val_loader_s1 = torch.utils.data.DataLoader(
        val_dataset_s1, 
        batch_size=args.batch_size_s1, 
        shuffle=False, 
        num_workers=args.num_workers,
        #pin_memory=True if device == 'cuda' else False
        pin_memory = False
    )
    
    train_loader_s2 = torch.utils.data.DataLoader(
        train_dataset_s2, 
        batch_size=args.batch_size_s2, 
        shuffle=True, 
        num_workers=args.num_workers,
        #pin_memory=True if device == 'cuda' else False,
        pin_memory = False,
        drop_last = True
    )
    
    val_loader_s2 = torch.utils.data.DataLoader(
        val_dataset_s2, 
        batch_size=args.batch_size_s2, 
        shuffle=False, 
        num_workers=args.num_workers,
        #pin_memory=True if device == 'cuda' else False,
        pin_memory = False,
        drop_last = True
    )
    
    # Print dataset information
    print_dataset_info(train_subjects, val_subjects, train_loader_s1, val_loader_s1, 
                      train_loader_s2, val_loader_s2)
    
    # ==========================================
    # STAGE 1: Single-view training
    # ==========================================
    
    stage1_log_dir = os.path.join(main_log_dir, 'stage_1')
    
    if not args.skip_stage1:
        print("\n" + "="*80)
        print("STAGE 1: SINGLE-VIEW TRAINING")
        print("="*80)
        
        create_folder(stage1_log_dir)
        
#        # Create Stage 1 model
#        pretraining = not args.no_pretraining
#        stage1_model = SVCNN_RGBD(
#            name="svcnn_rgbd",
#            nfeatures=output_dim,
#            cnn_name=args.cnn_name,
#            pretraining=pretraining
#        )
#        # Create the model with the correct number of outputs
#        stage1_model = SVCNN_RGBD(
#            name=f"svcnn_{args.model_type}",
#            nfeatures=nfeatures,
#            cnn_name=args.cnn_name,
#            pretraining=True
#        )
        
        # Create the ensemble model instead of a single one
        stage1_model = SVCNNEnsemble(
            num_models=3, # Train 3 models at once
            nfeatures=nfeatures,
            cnn_name=args.cnn_name
        )

#        # Add a Dropout layer to the model ---
#        # This replaces the final layer with a sequence containing Dropout
#        if hasattr(stage1_model.net, 'fc'):
#            num_ftrs = stage1_model.net.fc.in_features
#            stage1_model.net.fc = nn.Sequential(
#                nn.Dropout(p=0.5),
#                nn.Linear(num_ftrs, nfeatures)
#            )
#        print("Added Dropout layer before the final linear layer.")
            
        # Setup multi-GPU
        stage1_model = setup_multi_gpu(stage1_model, device)
        
        # Resume from checkpoint if specified
        if args.resume_stage1:
            print(f"Resuming Stage 1 from: {args.resume_stage1}")
            if torch.cuda.device_count() > 1:
                stage1_model.module.load_state_dict(torch.load(args.resume_stage1))
            else:
                stage1_model.load_state_dict(torch.load(args.resume_stage1))
        
        # Create optimizer and scheduler here ---
        #optimizer = optim.SGD(stage1_model.parameters(), lr=args.lr_s1, momentum=0.9, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(stage1_model.parameters(), lr=args.lr_s1, weight_decay=1e-4)
        
        # Configure for 3 cycles of 10 epochs each (total 30 epochs)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        
        stage1_trainer = create_stage1_trainer(
            model=stage1_model,
            train_loader=train_loader_s1,
            val_loader=val_loader_s1,
            optimizer=optimizer, # Pass the optimizer
            #scheduler=scheduler, # Pass the scheduler
            log_dir=stage1_log_dir,
            model_type=args.model_type,
            device=device
        )
        
        # Train Stage 1
        print(f"Starting Stage 1 training for {args.epochs_s1} epochs...")
        best_distance_s1 = stage1_trainer.train(args.epochs_s1)
        print(f"Stage 1 completed! Best validation distance: {best_distance_s1:.4f}")
        
        # Get the underlying model for Stage 2 initialization
        if torch.cuda.device_count() > 1:
            stage1_model_for_s2 = stage1_model.module
        else:
            stage1_model_for_s2 = stage1_model
            
    else:
        print("\n" + "="*80)
        print("STAGE 1: SKIPPED (loading from checkpoint)")
        print("="*80)
        
        # Load Stage 1 model for Stage 2 initialization
        stage1_model_for_s2 = SVCNN_RGBD(
            name="svcnn_rgbd",
            nfeatures=11,
            cnn_name=args.cnn_name,
            pretraining=False  # Doesn't matter since we're loading weights
        )
    # Load the state dict
    state_dict = torch.load(args.resume_stage1, map_location=device)

    # Remove 'module.' prefix from keys if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v

    # Load the modified state dict
    stage1_model_for_s2.load_state_dict(new_state_dict)
    print(f"Loaded Stage 1 model from: {args.resume_stage1}")
    
    # ==========================================
    # STAGE 2: Multi-view training
    # ==========================================
    
    print("\n" + "="*80)
    print("STAGE 2: MULTI-VIEW TRAINING (All Configurations)")
    print("="*80)
    print("Training on 3 configurations simultaneously:")
    print("  - 3 views (minimal)")
    print("  - 36 views (frontal)")
    print("  - 72 views (full)")
    
    stage2_log_dir = os.path.join(main_log_dir, 'stage_2')
    create_folder(stage2_log_dir)
    
    # Create Stage 2 model (initialized from Stage 1)
    stage2_model = AnthropometricViewGCN(
        name="anthropometric_viewgcn",
        svcnn_model=stage1_model_for_s2,
        nfeatures=output_dim,
        cnn_name=args.cnn_name
    )
    
    #stage2_model.to(device)
    # Setup multi-GPU
    stage2_model = setup_multi_gpu(stage2_model, device)
    
    # Resume from checkpoint if specified
    if args.resume_stage2:
        print(f"Resuming Stage 2 from: {args.resume_stage2}")
        if torch.cuda.device_count() > 1:
            stage2_model.module.load_state_dict(torch.load(args.resume_stage2))
        else:
            stage2_model.load_state_dict(torch.load(args.resume_stage2))
    
    # Create trainer
    stage2_trainer = create_stage2_trainer(
        model=stage2_model,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        config_name=args.config_name,
        task_type=args.task_type,  # NEW
        learning_rate=args.lr_s2,
        log_dir=stage2_log_dir,
        device=device
    )
    
    # Train Stage 2
    print(f"Starting Stage 2 training for {args.epochs_s2} epochs...")
    best_distance_s2 = stage2_trainer.train(args.epochs_s2)
    print(f"Stage 2 completed! Best validation distance: {best_distance_s2:.4f}")
    
    # ==========================================
    # Training Summary
    # ==========================================
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    
    if not args.skip_stage1:
        print(f"Stage 1 (Single-view):")
        print(f"  Best validation distance: {best_distance_s1:.4f}")
        print(f"  Model saved in: {stage1_log_dir}")
    
    print(f"Stage 2 (Multi-view):")
    print(f"  Best validation distance: {best_distance_s2:.4f}")
    print(f"  Model saved in: {stage2_log_dir}")
    
    print(f"\nExperiment logs: {main_log_dir}")
    print(f"Tensorboard: tensorboard --logdir {main_log_dir}")
    
    # Save final summary
    summary = {
        'experiment_name': args.name,
        'train_subjects': len(train_subjects),
        'val_subjects': len(val_subjects),
        'cnn_backbone': args.cnn_name,
        'stage1_best_distance': best_distance_s1 if not args.skip_stage1 else 'skipped',
        'stage2_best_distance': best_distance_s2,
        'total_epochs': (args.epochs_s1 if not args.skip_stage1 else 0) + args.epochs_s2
    }
    
    summary_path = os.path.join(main_log_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {summary_path}")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
