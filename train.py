import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import argparse
from models import RNNModel, LSTMModel, TransformerModel
from text_processor import TextProcessor, TextDataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

def setup_plotting_dir(base_dir):
    """Create plots directory if it doesn't exist."""
    plots_dir = base_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def plot_losses(train_losses, val_losses, plots_dir, model_type):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Model Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(plots_dir / f'{model_type}_loss_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_batch_losses(batch_losses, plots_dir, model_type, epoch):
    """Plot batch-level losses for each epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(batch_losses, label='Batch Loss', alpha=0.7)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Model - Epoch {epoch} Batch Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(plots_dir / f'{model_type}_batch_losses_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*20} {title} {'='*20}")

def train_epoch(model, data_loader, optimizer, criterion, device, epoch, plots_dir, model_type):
    model.train()
    total_loss = 0
    batches = 0
    batch_losses = []
    
    print_section(f"Training Epoch {epoch}")
    progress = tqdm(data_loader)
    
    for batch in progress:
        try:
            # Move data to device
            src = batch['src_tokens'].to(device)
            tgt = batch['tgt_tokens'].to(device)
            mask = batch['attn_mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, attention_mask=mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update stats
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss
            batches += 1
            avg_loss = total_loss / batches
            progress.set_description(f"Loss: {avg_loss:.4f}")
            
        except Exception as e:
            print(f"Warning: Error in batch: {e}")
    
    # Plot batch losses
    plot_batch_losses(batch_losses, plots_dir, model_type, epoch)
    
    return total_loss / batches if batches > 0 else float('inf')

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    batches = 0
    
    print("Validating...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            try:
                # Move data to device
                src = batch['src_tokens'].to(device)
                tgt = batch['tgt_tokens'].to(device)
                mask = batch['attn_mask'].to(device)
                
                # Forward pass
                output = model(src, attention_mask=mask)
                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                
                # Update stats
                total_loss += loss.item()
                batches += 1
                
            except Exception as e:
                print(f"Warning: Validation error: {e}")
    
    return total_loss / batches if batches > 0 else float('inf')

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = Path(args.base_dir)
    plots_dir = setup_plotting_dir(base_dir)
    
    print_section("Setup")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    
    # Load data
    print_section("Loading Data")
    processor = TextProcessor(base_dir)
    dataset = TextDataset(base_dir / 'processed/dataset.jsonl', processor.tokenizer, args.max_len)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    # Initialize model
    print_section("Model Initialization")
    vocab_size = processor.tokenizer.get_piece_size()
    model_args = {
        'vocab_size': vocab_size,
        'dim_embed': args.dim_embed,
        'dim_hidden': args.dim_hidden,
        'num_blocks': args.num_layers,
        'p_drop': args.p_drop
    }
    
    if args.model_type == 'transformer':
        model = TransformerModel(**model_args, n_heads=args.num_heads, seq_len_max=args.max_len)
    elif args.model_type == 'lstm':
        model = LSTMModel(**model_args)
    else:
        model = RNNModel(**model_args)
    
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=2, label_smoothing=0.1)
    
    # Training loop
    print_section("Training")
    best_loss = float('inf')
    patience = args.patience
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1, plots_dir, args.model_type)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Plot losses
        plot_losses(train_losses, val_losses, plots_dir, args.model_type)
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            save_path = base_dir / f'{args.model_type}_model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch': epoch,
            }, save_path)
            print(f"Model saved to {save_path}")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered")
                break
    
    # Save final loss curves
    plot_losses(train_losses, val_losses, plots_dir, f"{args.model_type}_final")

def main():
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--base_dir', type=str, default='.',
                      help='Base directory containing processed/ folder and tokenizer')
    parser.add_argument('--model_type', type=str, default='transformer',
                      choices=['transformer', 'lstm', 'rnn'])
    parser.add_argument('--dim_embed', type=int, default=384)
    parser.add_argument('--dim_hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--p_drop', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
