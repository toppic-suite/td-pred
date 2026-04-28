import numpy as np
import torch
from torch.utils.data import Dataset

import model.model_data as md
import h5py

class MsalignAnnoBatchGenerator(Dataset):
    def __init__(self, msalign_file, max_seq_length=200):
        # default parameters
        self.msalign_file = msalign_file
        self.max_seq_length = max_seq_length
        self.valid_spectra = list(self.read_msalign_iter(msalign_file))
        print(f"Total spectra: {len(self.valid_spectra)}")

    def __len__(self):
        return len(self.valid_spectra)

    def __getitem__(self, idx):
        spectrum = self.valid_spectra[idx]
        
        # preprocess single spectrum
        embedding = md.embed_spectrum(spectrum)
        meta = md.embed_meta(spectrum)
        mask = md.get_mask(spectrum, self.max_seq_length)    
        target = self.spectrum_anno_to_b_ion(spectrum)

        # Convert to PyTorch tensors
        embedding_tensor = torch.from_numpy(embedding)
        meta_tensor = torch.from_numpy(meta)
        mask_tensor = torch.from_numpy(mask)
        target_tensor = torch.from_numpy(target)

        return (embedding_tensor, meta_tensor, mask_tensor), target_tensor

    def read_msalign_iter(self, fn):
        spectra = self.parse_msalign_annot(fn)
        for spec in spectra:
            meta = spec["meta"]
            dataset = meta.get("DATASET_ID", "")
            mzml_filename = meta.get("MZML_FILE_NAME", "")
            scan = meta.get("MS2_SCAN", "")
            proteoform = meta.get("DATABASE_SEQUENCE", "")           
            if proteoform == "":
                continue
            charge_str = meta.get("PRECURSOR_CHARGE", "")
            charge_list = charge_str.split(":")
            prec_charge = int(charge_list[0])
            prec_mz_str = meta.get("PRECURSOR_MONOISOTOPIC_MZ", "0")
            prec_mz_list = prec_mz_str.split(":")
            prec_mz = float(prec_mz_list[0])
            nce_str = meta.get("COLLISION_ENERGY", "")
            if (nce_str == ""):
                nce = 0.0
            else:
                nce = float(nce_str)
            activation = meta.get("ACTIVATION", "unknown").lower()
            instrument = meta.get("INSTRUMENT", "").lower()

            frag_masses = spec["peaks"]
            mass_list = np.array([p[0] for p in frag_masses], dtype=np.float32)
            intensity_list = np.array([p[1] for p in frag_masses], dtype=np.float32)
            charge_list = np.array([p[2] for p in frag_masses], dtype=np.int32)
            ion_type_list = [p[3] for p in frag_masses]
            ion_pos_list = np.array([p[4] for p in frag_masses], dtype=np.int32)


            yield {
                "dataset": dataset,
                "mzml_filename": mzml_filename,
                "scan": scan,
                "instrument": instrument,
                "proteoform": proteoform,
                "prec_charge": prec_charge,
                "prec_mz": prec_mz,
                "activation_type": activation,
                "nce": nce,
                "frag_mass_list": mass_list,
                "frag_intensity_list": intensity_list,
                "frag_charge_list": charge_list,
                "frag_type_list": ion_type_list,
                "frag_pos_list": ion_pos_list,
            }

    def parse_msalign_annot(self, msalign_file):
        spectra = []
        with open(msalign_file, "r") as f:
            current = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("BEGIN IONS"):
                    current = {"meta": {}, "peaks": []}
                elif line.startswith("END IONS"):
                    if current:
                        spectra.append(current)
                        current = None
                elif "=" in line:
                    key, val = line.split("=", 1)
                    current["meta"][key] = val
                else:
                    # Each line should be: mz intensity charge ion_type ion_pos
                    parts = line.split()
                    if len(parts) == 10:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        ch = int(parts[2])
                        ion_type = parts[4].strip()[:1].lower()
                        ion_pos = int(parts[5])
                        current["peaks"].append((mz, intensity, ch, ion_type, ion_pos))
        return spectra

    def spectrum_anno(self, spectrum):
        #print("spectrum", spectrum)
        pep_bond_target = np.zeros((self.max_seq_length - 1), dtype=np.float32)
        b_y_target = np.zeros((self.max_seq_length - 1, 2), dtype=np.float32)
        max_charge = md.get_max_fragment_charge()
        charge_target = np.zeros((self.max_seq_length - 1, max_charge*2), dtype=np.int32)
        prec_charge = spectrum["prec_charge"]
        frag_inte_list = np.array(spectrum["frag_intensity_list"], dtype=np.float32)
        charge_list = np.array(spectrum["frag_charge_list"], dtype=np.float32)
        ion_type_list = np.array(spectrum["frag_type_list"])
        ion_type_positions = np.array(spectrum["frag_pos_list"], dtype=np.int32)
        seq_length = len(spectrum['proteoform'])
        for intensity, type, pos, charge in zip(frag_inte_list, ion_type_list, ion_type_positions, charge_list):
            #print("intensity, type, pos", intensity, type, pos)
            if charge > prec_charge:
                continue
            if type != "b" and type != "y" and type != "c" and type != "z":
                print("Warning: Unknown ion type:", type)
                continue
            if pos < 1 or pos >= seq_length:
                continue
            if type == "b" or type == "c":
                idx = pos -1
                pep_bond_target[idx] += intensity
                b_y_target[idx, 0] += intensity
                charge_target[idx, charge - 1] += intensity
            elif type == "y" or type == "z":
                idx = seq_length - pos - 1
                pep_bond_target[idx] += intensity
                b_y_target[idx, 1] += intensity
                charge_target[idx, max_charge + charge - 1] += intensity
        # Normalize intensities by max
        inte_max = pep_bond_target.max() if pep_bond_target.max() > 0 else 1.0
        pep_bond_target = pep_bond_target / inte_max
        b_y_target = b_y_target.flatten()  # flatten to 1D
        inte_max = b_y_target.max() if b_y_target.max() > 0 else 1.0
        b_y_target = b_y_target / inte_max
        charge_target = charge_target.flatten()
        inte_max = charge_target.max() if charge_target.max() > 0 else 1.0
        charge_target = charge_target / inte_max
        return pep_bond_target, b_y_target, charge_target
    
    def convert_anno_msalign_to_hdf5(self, hdf5_file):
        num_spectra = len(self.valid_spectra)
        seq_encoding_length = md.get_seq_encoding_length(self.max_seq_length)
        seq_encoding_dimension = md.get_seq_encoding_dimension()
        meta_length = md.get_meta_length()
        max_charge = md.get_max_fragment_charge()
        print(f"Converting {num_spectra} valid spectra to HDF5 format")
        print(f"Creating HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, 'w') as h5f:
            # Create datasets
            id_dset = h5f.create_dataset(
                'spectrum_id',
                shape=(num_spectra,1),
                dtype = "int32"
            )      
            scan_dset = h5f.create_dataset(
                'scan',
                shape=(num_spectra, md.get_scan_info_length()),
                dtype = h5py.string_dtype()
            )
            seq_encoding_dset = h5f.create_dataset(
                'seq_encoding',
                shape=(num_spectra, seq_encoding_length, seq_encoding_dimension),
                dtype='float32'
            )
            meta_dset = h5f.create_dataset(
                'meta',
                shape=(num_spectra, meta_length),
                dtype='float32'
            )
            mask_dset = h5f.create_dataset(
                'mask',
                shape=(num_spectra, seq_encoding_length),
                dtype='float32'
            )
            charge_mask_dset = h5f.create_dataset(
                'charge_mask',
                shape=(num_spectra, max_charge*2),
                dtype='float32'
            )   
            pep_bond_target_dset = h5f.create_dataset(
                'pep_bond_target',
                shape=(num_spectra, self.max_seq_length - 1), 
                dtype='float32'
            )
            b_y_target_dset = h5f.create_dataset(
                'b_y_target',
                shape=(num_spectra, (self.max_seq_length - 1) * 2),  # Assuming flattening doubles the length
                dtype='float32'
            )
            charge_target_dset = h5f.create_dataset(
                'charge_target',
                shape=(num_spectra, (self.max_seq_length - 1) * (2 * max_charge)),  # Assuming flattening and charge dimension
                dtype='float32'
            )

            # Process and store each spectrum
            for idx in range(num_spectra):
                if idx % 1000 == 0:
                    print(f"Processing spectrum {idx}/{num_spectra}")
                spectrum = self.valid_spectra[idx]
                scan = md.get_scan_info(spectrum)
                seq_encoding = md.encode_spectrum(spectrum, self.max_seq_length)
                meta = md.encode_meta(spectrum)
                mask = md.get_mask(spectrum, self.max_seq_length)
                prec_charge = int(spectrum["prec_charge"])
                charge_mask = md.get_charge_mask(prec_charge)
                pep_bond_target, b_y_target, charge_target = self.spectrum_anno(spectrum)
                if idx == 0:
                    print("The first target:", pep_bond_target, "sum", pep_bond_target.sum())
                # Store in HDF5
                id_dset[idx] = idx
                scan_dset[idx] = scan
                seq_encoding_dset[idx] = seq_encoding
                meta_dset[idx] = meta
                mask_dset[idx] = mask
                charge_mask_dset[idx] = charge_mask
                pep_bond_target_dset[idx] = pep_bond_target
                b_y_target_dset[idx] = b_y_target
                charge_target_dset[idx] = charge_target
        print(f"Successfully converted {idx+1} spectra records to {hdf5_file}")