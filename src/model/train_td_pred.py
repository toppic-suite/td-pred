#!/usr/bin/env python3

import os
import numpy as np
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import model_data as md
import td_pred_model
import hdf5_generator


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1

def is_main_process():
    return get_rank() == 0


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed_value, deterministic=False):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True lets CuDNN auto-tune kernels for the fixed input sizes
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_datasets(train_file, val_file, target_type="pep_bond"):
    if is_main_process():
        print(f"Loading datasets from hdf5 files...")
    train_dataset = hdf5_generator.Hdf5BatchGenerator(train_file, target_type=target_type)
    val_dataset = hdf5_generator.Hdf5BatchGenerator(val_file, target_type=target_type)
    return train_dataset, val_dataset


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_model(model, train_dataloader, val_dataloader, num_epochs=40,
                learning_rate=0.0003, device='cpu', model_file="model.hdf",
                use_amp=True, train_sampler=None):
    start_time = time.time()

    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # GradScaler is a no-op when enabled=False (CPU or --no_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_losses = []
    train_similarities = []
    val_losses = []
    val_similarities = []

    if is_main_process():
        print(f"Starting training for {num_epochs} epochs on device: {device}")
        if get_world_size() > 1:
            print(f"Distributed training across {get_world_size()} GPUs (DDP)")
        if use_amp:
            print("Automatic Mixed Precision (AMP) enabled")
        print("-" * 50)

    best_val_similarity = 0.0
    round_without_improvement = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Required so each epoch gets a different shuffle across DDP ranks
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ------------------------------------------------------------------ #
        # Training phase
        # ------------------------------------------------------------------ #
        model.train()
        train_loss_sum = torch.tensor(0.0, device=device)
        train_sim_sum  = torch.tensor(0.0, device=device)
        train_batches  = torch.tensor(0,   device=device)

        for (spectrum_id, seq_encoding, meta, mask, charge_mask), target in train_dataloader:
            seq_encoding = seq_encoding.to(device, non_blocking=True)
            meta         = meta.to(device, non_blocking=True)
            mask         = mask.to(device, non_blocking=True)
            charge_mask  = charge_mask.to(device, non_blocking=True)
            target       = target.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                output = model(seq_encoding, meta, mask, charge_mask)
                loss   = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.detach()
            train_sim_sum  += F.cosine_similarity(output.detach(), target, dim=1).mean()
            train_batches  += 1

        # Sum metrics across all DDP ranks then divide
        if is_dist_initialized():
            for t in (train_loss_sum, train_sim_sum, train_batches):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        avg_train_loss       = (train_loss_sum / train_batches).item()
        avg_train_similarity = (train_sim_sum  / train_batches).item()
        train_losses.append(avg_train_loss)
        train_similarities.append(avg_train_similarity)

        # ------------------------------------------------------------------ #
        # Validation phase  (all ranks evaluate the full val set independently;
        # metrics are identical on every rank so no all_reduce needed here)
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_sim_sum  = torch.tensor(0.0, device=device)
        val_batches  = torch.tensor(0,   device=device)

        val_spectrum_id_list = []
        val_similarity_list  = []

        with torch.no_grad():
            for (spectrum_id, seq_encoding, meta, mask, charge_mask), target in val_dataloader:
                seq_encoding = seq_encoding.to(device, non_blocking=True)
                meta         = meta.to(device, non_blocking=True)
                mask         = mask.to(device, non_blocking=True)
                charge_mask  = charge_mask.to(device, non_blocking=True)
                target       = target.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    output = model(seq_encoding, meta, mask, charge_mask)
                    loss   = criterion(output, target)

                cosine_sim = F.cosine_similarity(output, target, dim=1)
                val_loss_sum += loss
                val_sim_sum  += cosine_sim.mean()
                val_batches  += 1

                if is_main_process():
                    val_spectrum_id_list.append(spectrum_id.squeeze(1))
                    val_similarity_list.append(cosine_sim.cpu())

        avg_val_loss       = (val_loss_sum / val_batches).item()
        avg_val_similarity = (val_sim_sum  / val_batches).item()
        val_losses.append(avg_val_loss)
        val_similarities.append(avg_val_similarity)

        epoch_duration = time.time() - epoch_start_time

        if is_main_process():
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Training Loss: {avg_train_loss:.6f}, "
                  f"Training Cosine Similarity: {avg_train_similarity:.6f}, "
                  f"Validation Loss: {avg_val_loss:.6f}, "
                  f"Validation Cosine Similarity: {avg_val_similarity:.6f}, "
                  f"Time: {epoch_duration:.2f}s")

            similarity_filename = "similarity_" + str(epoch+1) + ".tsv"
            with open(similarity_filename, 'w') as f:
                f.write("spectrum_id\tsimilarity\n")
                all_ids  = torch.cat(val_spectrum_id_list)
                all_sims = torch.cat(val_similarity_list)
                for spec_id, sim in zip(all_ids, all_sims):
                    f.write(f"{spec_id.item()}\t{sim.item()}\n")

            # Unwrap DDP/DataParallel to save bare model weights
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'model_state_dict': state_dict,
                'train_losses': train_losses,
                'train_similarities': train_similarities,
                'val_losses': val_losses,
                'val_similarities': val_similarities,
            }, model_file + "_" + str(epoch+1))

        if avg_val_similarity > best_val_similarity:
            best_val_similarity = avg_val_similarity
            round_without_improvement = 0
        else:
            round_without_improvement += 1
        #if round_without_improvement >= 10:
        #    print("No improvement in validation similarity for 10 consecutive epochs. Stopping training early.")
        #    break

    total_time = time.time() - start_time
    return train_losses, train_similarities, val_losses, val_similarities, total_time


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Device / distributed setup
#
# Single-GPU / CPU:  python train_td_pred.py ...
# Multi-GPU DDP:     torchrun --nproc_per_node=N train_td_pred.py ...
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      type=str,   default="hcd_train_dataset.mgf",      help="train data filename")
    parser.add_argument("--validate",   type=str,   default="hcd_validation_dataset.mgf", help="validation data filename")
    parser.add_argument("--out",        type=str,   default="pred_full_torch_model.pth",  help="filename to save the trained model")
    parser.add_argument("--batch",      type=int,   default=32,    help="Batch size")
    parser.add_argument("--epochs",     type=int,   default=50,    help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=0.0003,help="Learning rate")
    parser.add_argument("--load_model", type=str,   default=None,  help="Path to presaved model to load")
    parser.add_argument("--max_length", type=int,   default=200,   help="Length of input sequences (default: 200)")
    parser.add_argument("--target",     type=str,   default="pep_bond", help="Target type: pep_bond or b_y or charge")
    parser.add_argument("--seed",       type=int,   default=42,    help="Seed")
    parser.add_argument("--no_amp",     action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic algorithms for reproducibility (disables cudnn.benchmark, slower)")
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank >= 0:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if is_main_process():
        print(f"Using device: {device}")
        if get_world_size() > 1:
            print(f"World size: {get_world_size()} processes")

    # Each rank gets a different seed so dropout masks are independent across GPUs.
    # DDP broadcasts rank 0's weights at startup, so model init is still identical.
    effective_seed = args.seed + get_rank()
    print(f"Seed: {args.seed} (effective: {effective_seed} on rank {get_rank()})")
    set_seed(effective_seed, deterministic=args.deterministic)

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    output_len = args.max_length - 1
    if args.target == "pep_bond":
        output_dim = 1
    elif args.target == "b_y":
        output_dim = 2
    elif args.target == "charge":
        output_dim = md.get_max_fragment_charge() * 2
    else:
        print("Using default target: pep_bond")
        output_dim = 1

    single_model = td_pred_model.TransformerSeq2Seq(
        output_len=output_len,
        output_dim=output_dim,
        max_seq_length=args.max_length,
    )

    if args.load_model:
        if is_main_process():
            print(f"Loading presaved model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location='cpu')
        single_model.load_state_dict(checkpoint['model_state_dict'])
        if is_main_process():
            print("Model loaded successfully")

    single_model.to(device)

    if local_rank >= 0:
        model = DDP(single_model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1:
        if is_main_process():
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(single_model)
    else:
        model = single_model

    if is_main_process():
        n_params = sum(p.numel() for p in single_model.parameters())
        print(f"Total parameters:     {n_params:,}")
        print(f"Trainable parameters: {sum(p.numel() for p in single_model.parameters() if p.requires_grad):,}")

        # Sanity-check forward pass
        batch_size = 4
        seq_enc_in     = torch.randn(batch_size, md.get_seq_encoding_length(args.max_length),
                                           md.get_seq_encoding_dimension()).to(device)                                                                           
        meta_in        = torch.randn(batch_size, md.get_meta_length()).to(device)                                                                          
        mask_in        = torch.randn(batch_size, md.get_seq_encoding_length(args.max_length)).to(device)                                                   
        charge_mask_in = torch.randn(batch_size, 2 * md.get_max_fragment_charge()).to(device)   
        print("seq_encoding_input shape:", seq_enc_in.shape, " meta_input shape:", meta_in.shape)
        with torch.no_grad():
            out = single_model(seq_enc_in, meta_in, mask_in, charge_mask_in)
        print(f"Output shape: {out.shape}  (expected: ({batch_size}, {single_model.output_len}))")

    # ------------------------------------------------------------------ #
    # Data loaders
    # ------------------------------------------------------------------ #
    train_dataset, val_dataset = load_datasets(args.train, args.validate, target_type=args.target)

    if is_main_process():
        print(f"Training dataset size:   {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist_initialized() else None

    num_workers = min(8, os.cpu_count() or 2)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        **loader_kwargs,
    )

    use_amp = not args.no_amp and torch.cuda.is_available()

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    train_losses, train_similarities, val_losses, val_similarities, training_time = train_model(
        model, train_dataloader, val_dataloader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        model_file=args.out,
        use_amp=use_amp,
        train_sampler=train_sampler,
    )

    # ------------------------------------------------------------------ #
    # Save final model (rank 0 only)
    # ------------------------------------------------------------------ #
    if is_main_process():
        print(f"\nSaving model to {args.out}")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'model_state_dict': state_dict,
            'train_losses': train_losses,
            'train_similarities': train_similarities,
            'val_losses': val_losses,
            'val_similarities': val_similarities,
            'training_time': training_time,
        }, args.out + "_final")

        hours   = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = training_time % 60
        print("Training completed successfully!")
        print(f"Final training loss:   {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Total training time:   {hours}h {minutes}m {seconds:.2f}s ({training_time:.2f} seconds)")

    if is_dist_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
