#!/usr/bin/env python3

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class Hdf5BatchGenerator(Dataset):
    # PyTorch Dataset for reading preprocessed spectra from HDF5 file
    def __init__(self, hdf5_file, target_type="pep_bond"):
        self.hdf5_file = hdf5_file
        self.target_type = 0
        if (target_type == "pep_bond"):
            self.target_type = 0
        elif (target_type == "b_y"):
            self.target_type = 1
        elif (target_type == "charge"):
            self.target_type = 2
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        self.h5f = h5py.File(hdf5_file, 'r')

        self.num_spectra = self.h5f['seq_encoding'].shape[0]
        self.spectrum_id_shape = self.h5f['spectrum_id'].shape[1:]
        self.seq_encoding_shape = self.h5f['seq_encoding'].shape[1:]
        self.meta_shape = self.h5f['meta'].shape[1:]
        self.mask_shape = self.h5f['mask'].shape[1:]
        self.pep_bond_target_shape = self.h5f['pep_bond_target'].shape[1:]
        self.b_y_target_shape = self.h5f['b_y_target'].shape[1:]
        self.charge_target_shape = self.h5f['charge_target'].shape[1:]

        print(f"Loaded HDF5 dataset with {self.num_spectra} spectra")
        print(f"Spectrum ID shape: {self.spectrum_id_shape}")
        print(f"Sequence encoding shape: {self.seq_encoding_shape}")
        print(f"Meta shape: {self.meta_shape}")
        print(f"Mask shape: {self.mask_shape}")
        print(f"Pep bond target shape: {self.pep_bond_target_shape}")
        print(f"B/Y target shape: {self.b_y_target_shape}")
        print(f"Charge target shape: {self.charge_target_shape}")

    def __len__(self):
        return self.num_spectra

    #  Get a single spectrum by index
    def __getitem__(self, idx):
        # Open file for each access (thread-safe for DataLoader with multiple workers)
        spectrum_id = self.h5f['spectrum_id'][idx]
        seq_encoding = self.h5f['seq_encoding'][idx]
        meta = self.h5f['meta'][idx]
        mask = self.h5f['mask'][idx]
        charge_mask = self.h5f['charge_mask'][idx]
        #print("charge_mask:", charge_mask)
        if self.target_type == 0:
            target = self.h5f['pep_bond_target'][idx]
        elif self.target_type == 1:
            target = self.h5f['b_y_target'][idx]
        elif self.target_type == 2:
            target = self.h5f['charge_target'][idx]
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")
        # Convert to PyTorch tensors
        spectrum_id = torch.from_numpy(spectrum_id)
        seq_encoding_tensor = torch.from_numpy(seq_encoding)
        meta_tensor = torch.from_numpy(meta)
        target = torch.from_numpy(target)
        mask_tensor = torch.from_numpy(mask)
        charge_mask_tensor = torch.from_numpy(charge_mask)

        return (spectrum_id, seq_encoding_tensor, meta_tensor, mask_tensor, charge_mask_tensor), target