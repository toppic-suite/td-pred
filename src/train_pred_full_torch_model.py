#!/usr/bin/env python3

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
#import pred_full_long_torch_model 
import pred_full_long_transformer_model
import model_data as md
import mgf_torch_generator
import hdf5_batch_generator
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.nn.functional as F
import time


def load_datasets(train_file, val_file, target_type="pep_bond"):
    """
    Load training and validation datasets based on file type

    Args:
        train_file: Path to training data file
        val_file: Path to validation data file

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    print(f"Loading datasets from hdf5 files...")

    train_dataset = hdf5_batch_generator.Hdf5BatchGenerator(train_file, target_type=target_type)
    val_dataset = hdf5_batch_generator.Hdf5BatchGenerator(val_file, target_type=target_type) 

    return train_dataset, val_dataset


def train_model(model, train_dataloader, val_dataloader, num_epochs=40, learning_rate=0.0003, device='cpu', model_file="model.hdf"):
    """Train the PyTorch model using MSE loss"""

    # Start training timer
    start_time = time.time()

    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_similarities = []
    val_losses = []
    val_similarities = []

    print(f"Starting training for {num_epochs} epochs on device: {device}")
    print("-" * 50)

    best_val_similarity = 0.0
    round_without_improvement = 0

    for epoch in range(num_epochs):
        # Start epoch timer
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_similarity = 0.0
        
        for batch_idx, ((spectrum_id, seq_encoding, meta, mask, charge_mask), target) in enumerate(train_dataloader):
            #print(f"train - Batch {batch_idx}: embedding size: {embedding.size()}, meta size: {meta.size()}")
            # Move data to device
            seq_encoding = seq_encoding.to(device)
            meta = meta.to(device)
            mask = mask.to(device)
            charge_mask = charge_mask.to(device)
            target = target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(seq_encoding, meta, mask, charge_mask)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1

            cosine_sim = F.cosine_similarity(output, target, dim=1).mean().item()
            train_similarity += cosine_sim
        
        avg_train_loss = train_loss / train_batches
        avg_train_similarity = train_similarity / train_batches
        train_losses.append(avg_train_loss)
        train_similarities.append(avg_train_similarity)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_similarity = 0.0

        val_spectrum_id_list = torch.empty(0, dtype=torch.int32)
        val_similarity_list = torch.empty(0) 
        with torch.no_grad():
            for batch_idx, ((spectrum_id, seq_encoding, meta, mask, charge_mask), target) in enumerate(val_dataloader):
                val_spectrum_id_list = torch.cat((val_spectrum_id_list, spectrum_id.squeeze(1)), dim=0)  
                # Move data to device
                seq_encoding = seq_encoding.to(device)
                meta = meta.to(device)
                mask = mask.to(device)  
                charge_mask = charge_mask.to(device)
                target = target.to(device)
                
                # Forward pass
                output = model(seq_encoding, meta, mask, charge_mask)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1
                cosine_sim = F.cosine_similarity(output, target, dim=1)
                val_similarity_list = torch.cat((val_similarity_list, cosine_sim.cpu()), dim=0)
                val_similarity += cosine_sim.mean().item()
        
        avg_val_loss = val_loss / val_batches
        avg_val_similarity = val_similarity / val_batches
        val_losses.append(avg_val_loss)
        val_similarities.append(avg_val_similarity)

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Training Loss: {avg_train_loss:.6f}, "
              f"Training Cosine Similarity: {avg_train_similarity:.6f}, "
              f"Validation Loss: {avg_val_loss:.6f}, "
              f"Validation Cosine Similarity: {avg_val_similarity:.6f}, "
              f"Time: {epoch_duration:.2f}s")

        # Write validation similarities to TSV file
        similarity_filename = "similarity_" + str(epoch+1) + ".tsv"
        with open(similarity_filename, 'w') as f:
            f.write("spectrum_id\tsimilarity\n")
            for spec_id, sim in zip(val_spectrum_id_list, val_similarity_list):
                f.write(f"{spec_id.item()}\t{sim.item()}\n")

        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'train_similarities': train_similarities,
            'val_losses': val_losses,
            'val_similarities': val_similarities,
        }, model_file + "_" + str(epoch+1))
        # Early stopping based on validation similarity
        if avg_val_similarity > best_val_similarity:
            best_val_similarity = avg_val_similarity
            round_without_improvement = 0   
        else:
            round_without_improvement += 1
        if round_without_improvement >= 10:
            print("No improvement in validation similarity for 10 consecutive epochs. Stopping training early.")
            break

    # Calculate total training time
    end_time = time.time()
    total_time = end_time - start_time

    return train_losses, train_similarities, val_losses, val_similarities, total_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, help="train data filename", default="hcd_train_dataset.mgf"
    )
    parser.add_argument(
        "--validate", type=str, help="validation data filename", default="hcd_validation_dataset.mgf"
    )
    parser.add_argument(
        "--out", type=str, help="filename to save the trained model", default="pred_full_torch_model.pth"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--load_model", type=str, default=None, help="Path to presaved model to load")
    parser.add_argument("--max_length", type=int, default=200, help="Length of input sequences (default: 200)")
    parser.add_argument("--target", type=str, default="pep_bond", help="Target type: pep_bond or b_y or charge")
    args = parser.parse_args()

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    output_len = args.max_length -1
    output_dim = 1
    if args.target == "pep_bond":
        output_dim = 1
    elif args.target == "b_y":
        output_dim = 2
    elif args.target == "charge":
        max_charge = md.get_max_fragment_charge()
        output_dim = max_charge * 2
    else:
        print("Using default target: pep_bond")
        output_dim = 1
    single_model = pred_full_long_transformer_model.TransformerSeq2Seq(
        output_len=output_len,
        output_dim=output_dim,
        max_seq_length=args.max_length,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(single_model)
    else:
        model = single_model

    # Load presaved model if specified
    if args.load_model:
        print(f"Loading presaved model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with dummy inputs
    batch_size = 4
    seq_encoding_input = torch.randn(batch_size, md.get_seq_encoding_length(args.max_length), 
                                     md.get_seq_encoding_dimension())
    meta_input = torch.randn(batch_size, md.get_meta_length())
    mask_input = torch.randn(batch_size, md.get_seq_encoding_length(args.max_length))
    charge_mask_input = torch.randn(batch_size, 2 * md.get_max_fragment_charge())
    print("embedding_input shape:", seq_encoding_input.shape, "meta_input shape:", meta_input.shape)

    #summary(model, input_data = (embedding_input, meta_input))
    
    with torch.no_grad():
        output = single_model(seq_encoding_input, meta_input, mask_input, charge_mask_input)
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, {single_model.output_len})")

    # Create data generators
    train_dataset, val_dataset = load_datasets(args.train, args.validate, target_type=args.target)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2)

    # Train model
    train_losses, train_similarities, val_losses, val_similarities, training_time = train_model(
        model, train_dataloader, val_dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        model_file=args.out
    )

    # Save the trained model
    print(f"\nSaving model to {args.out}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_similarities': train_similarities,
        'val_losses': val_losses,
        'val_similarities': val_similarities,
        'training_time': training_time,
    }, args.out + "_final")

    # Convert training time to hours, minutes, seconds
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = training_time % 60

    print("Training completed successfully!")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Total training time: {hours}h {minutes}m {seconds:.2f}s ({training_time:.2f} seconds)")

if __name__ == "__main__":
    main()
