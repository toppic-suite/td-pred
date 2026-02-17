import numpy as np
import argparse
import torch
from torch.utils.data import Dataset
import h5py
import re

from pyteomics import mgf, mass

class MgfBatchGenerator(Dataset):
    def __init__(self, mgf_file, max_peptide_length=200):
        # default parameters
        self.MAX_PEPTIDE_LENGTH = max_peptide_length
        self.INPUT_LENGTH = self.MAX_PEPTIDE_LENGTH + 2
        self.META_SHAPE = (3, 30)
        self.BIN_SIZE = 0.1
        self.SPECTRA_DIMENSION = 20
        self.LENGTH_SCALE = 1000
        self.PRECURSOR_SCALE = 20000.0
        self.mono_mass_list = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032029,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.04768,
            "C": 103.00918,
            "L": 113.08406,
            "I": 113.08406,
            "D": 115.02694,
            "Q": 128.05858,
            "K": 128.09496,
            "E": 129.04259,
            "M": 131.04048,
            "m": 147.0354,
            "H": 137.05891,
            "F": 147.06441,
            "R": 156.10111,
            "Y": 163.06333,
            "N": 114.04293,
            "W": 186.07931,
            "O": 147.03538,
            "U": 150.95363,
        }

        # init char map
        Alist = list("ACDEFGHIKLMNPQRSTUVWYZ")
        self.charMap = {"@": 0, "[": 22}
        for i, a in enumerate(Alist):
            self.charMap[a] = i + 1
        self.charMap["+"] = max(self.charMap.values()) + 1
        self.ENCODE_DIMENSION = len(Alist) + 3

        # amino acid + ending + mass + position + padding 
        self.INPUT_DIMENSION = self.ENCODE_DIMENSION + 2 + 3  

        self.mgf_file = mgf_file
        self.spectra = list(self.readmgf_iter(mgf_file))
        # Filter out invalid spectra at initialization
        self.valid_spectra = [sp for sp in self.spectra if len(sp["pep"]) <= self.MAX_PEPTIDE_LENGTH]

    def __len__(self):
        return len(self.valid_spectra)

    def __getitem__(self, idx):
        spectrum = self.valid_spectra[idx]
        
        # preprocess single spectrum
        embedding = self.embed_spectrum(spectrum)
        meta = self.embed_meta(spectrum)
        # compute y
        # y = self.spectrum2vector(spectrum["mz"], spectrum["it"], spectrum["mass"], self.BIN_SIZE, spectrum["charge"])
        y = self.spectrum2vector_2000(spectrum)

        # Convert to PyTorch tensors
        embedding_tensor = torch.from_numpy(embedding)
        meta_tensor = torch.from_numpy(meta)
        y_tensor = torch.from_numpy(y)

        return (embedding_tensor, meta_tensor), y_tensor

    def embed_spectrum(self, spectrum)  :
        embedding = np.zeros((self.INPUT_LENGTH, self.INPUT_DIMENSION), dtype="float32")
        self.embed(spectrum, embedding=embedding)
        return embedding
    
    def embed_meta(self, spectrum):
        meta = np.zeros(self.META_SHAPE, dtype="float32")
        meta[0][spectrum["charge"] - 1] = 1  # charge
        meta[1][spectrum["type"]] = 1  # ftype
        meta[2][0] = self.safe_fastmass(spectrum["pep"], ion_type="M", charge=1) / self.PRECURSOR_SCALE

        if not "nce" in spectrum or spectrum["nce"] == 0:
            meta[2][-1] = 0.25
        else:
            meta[2][-1] = spectrum["nce"] / 100.0
        return meta

    # def readmgf_iter(self, fn, type="hcd"):
    #     spec_types = {"unknown":0,"cid":1,"etd":2,"hcd":3,"ethcd":4,"etcid":5}
    #     with open(fn,"r") as f:
    #         for sp in mgf.read(f, convert_arrays=1, read_charges=False, dtype="float32", use_index=False):
    #             parsed = self.parse_spectra([sp], spec_type=spec_types[type])
    #             yield parsed[0]

    def readmgf_iter(self, fn, type="hcd"):
        spec_types = {"unknown":0,"cid":1,"etd":2,"hcd":3,"ethcd":4,"etcid":5}
        spectra = self.parse_mgf_annot(fn)
        for spec in spectra:
            meta = spec["meta"]
            peaks = spec["peaks"]

            mz = np.array([p[0] for p in peaks], dtype=np.float32)
            it = np.array([p[1] for p in peaks], dtype=np.float32)
            ch = np.array([p[2] for p in peaks], dtype=np.int32)
            labels = [p[3] for p in peaks]

            pep = meta.get("SEQ", meta.get("TITLE", ""))
            charge = int(str(meta.get("CHARGE", "1")).rstrip("+"))
            pepmass = float(meta.get("PEPMASS", 0).split()[0])
            nce = float(meta.get("ENERGY", 0))

            yield {
                "pep": pep,
                "charge": charge,
                "mass": pepmass,
                "mz": mz,
                "it": it,
                "fch": ch,
                "labels": labels,
                "nce": nce,
                "type": spec_types.get(type, 3),
            }


    def spectrum2vector(self, mz_list, itensity_list, mass, bin_size, charge):
        itensity_list = itensity_list / np.max(itensity_list)
        vector = np.zeros(self.SPECTRA_DIMENSION, dtype="float32")

        mz_list = np.asarray(mz_list)

        indexes = mz_list / bin_size
        indexes = np.around(indexes).astype("int32")

        for i, index in enumerate(indexes):
            if index >= self.SPECTRA_DIMENSION:
                continue

            vector[index] += itensity_list[i]

        # normalize
        vector = np.sqrt(vector)

        # remove precursors, including isotropic peaks
        for delta in (0, 1, 2):
            precursor_mz = mass + delta / charge
            if precursor_mz > 0 and precursor_mz < 2000:
                vector[round(precursor_mz / bin_size)] = 0

        return vector

    def safe_fastmass(self, pep, ion_type="M", charge=1):
        # Only keep standard amino acids
        valid_residues = set(mass.std_aa_mass.keys())
        clean_pep = "".join([aa for aa in pep if aa in valid_residues])

        if not clean_pep:
            # If all residues are invalid, return dummy mass
            return 0.0

        try:
            return mass.fast_mass(clean_pep, ion_type=ion_type, charge=charge)
        except Exception as e:
            print(f"Could not calculate mass for peptide {pep}: {e}")
            return 0.0

    # read inputs
    def parse_spectra(self, sps, spec_type=3):
        # ratio constants for NCE
        cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

        db = []

        for sp in sps:
            param = sp["params"]

            c = int(str(param["charge"][0])[0])

            if "seq" in param:
                pep = param["seq"]
            else:
                pep = param["title"]

            if "pepmass" in param:
                mass = param["pepmass"][0]
            else:
                mass = float(param["parent"])

            if "hcd" in param:
                try:
                    hcd = param["hcd"]
                    if hcd[-1] == "%":
                        hcd = float(hcd)
                    elif hcd[-2:] == "eV":
                        hcd = float(hcd[:-2])
                        hcd = hcd * 500 * cr[c] / mass
                    else:
                        raise Exception("Invalid type!")
                except:
                    hcd = 0
            else:
                hcd = 0

            mz = sp["m/z array"]
            it = sp["intensity array"]
            # Check if annotated peaks exist
            labels = []
            if "annotation" in sp:
                labels = sp["annotation"]  # optional field in some mgf variants
            elif "peaks" in sp and all(len(p) == 3 for p in sp["peaks"]):
                # e.g. (mz, intensity, ion_type)
                labels = [p[2] for p in sp["peaks"]]
            else:
                labels = [""] * len(mz)

            db.append(
                {
                    "pep": pep,
                    "charge": c,
                    "mass": mass,
                    "mz": mz,
                    "it": it,
                    "labels": labels,
                    "nce": hcd,
                    "type": spec_type,
             }
            )
        return db
    

    def parse_mgf_annot(self, mgf_file):
        spectra = []
        with open(mgf_file, "r") as f:
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
                    # Each line should be: mz intensity ion_type
                    parts = line.split()
                    if len(parts) == 4:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        ch = int(parts[2])
                        ion_type = parts[3].strip()
                        current["peaks"].append((mz, intensity, ch, ion_type))
        return spectra
    

    def spectrum2vector_2000(self, spectrum, seq_length=50, max_charge=20):
        """
        Convert annotated peaks (b/y ions) into a 2000-dimension vector.
        """
        v2000 = np.zeros((max_charge * 2, seq_length), dtype=np.float32)
        v2000_ion_type = np.zeros((max_charge * 2, seq_length), dtype=np.int32)
        v2000_ion_pos = np.zeros((max_charge * 2, seq_length), dtype=np.int32)

        # Each peak in spectrum["peaks"] should be (mz, intensity, label)
        mz_list = np.array(spectrum["mz"], dtype=np.float32)
        it_list = np.array(spectrum["it"], dtype=np.float32)
        ch_list = np.array(spectrum["fch"], dtype=np.int32)
        ann_list = np.array(spectrum["labels"])

        # charge = int(spectrum["charge"]) if "charge" in spectrum else 1
        
        for mz, intensity, charge, label in zip(mz_list, it_list, ch_list, ann_list):
            # m = re.match(r'([by])(\d+)', label)
            # if not m:
                # continue
            if label.startswith("b"):
                ion_type = "b"
                idx = int(label[1:])
            elif label.startswith("y"):
                ion_type = "y"
                idx = int(label[1:])
            else:
                continue
            # ion_type, idx_str = m.groups()
            # idx = int(idx_str)
            if charge > max_charge or idx < 1 or idx > seq_length:
                continue

            if ion_type == "b":
                v2000[charge - 1, idx - 1] = intensity
                v2000_ion_type[charge-1, idx -1] = 1 
                v2000_ion_pos[charge-1, idx - 1] = idx
            elif ion_type == "y":
                v2000[max_charge + charge - 1, seq_length - idx] = intensity
                v2000_ion_type[max_charge + charge - 1, seq_length - idx] = 2
                v2000_ion_pos[max_charge + charge - 1, seq_length - idx] = idx


        # Normalize intensities by max
        v2000_flat = v2000.flatten()
        max_i = v2000_flat.max() if v2000_flat.max() > 0 else 1.0
        v2000 = v2000 / max_i

        v2000_flat = v2000.T.flatten()
        v2000_ion_type_flat = v2000_ion_type.T.flatten()
        v2000_ion_pos_flat = v2000_ion_pos.T.flatten() 

        return v2000_flat, v2000_ion_type_flat, v2000_ion_pos_flat


    # embed input item into a matrix
    def embed(self, spectrum, embedding, mass_scale=200):
        pep = spectrum["pep"]
        pep = pep.replace("L", "I")

        embedding[len(pep)][self.ENCODE_DIMENSION - 1] = 1  # ending pos
        for i, aa in enumerate(pep):
            if aa not in self.charMap:
                continue
            embedding[i][self.charMap[aa]] = 1  # 1 - 20
            embedding[i][self.ENCODE_DIMENSION] = self.mono_mass_list.get(aa,0) / mass_scale

        embedding[: len(pep), self.ENCODE_DIMENSION + 1] = (
            np.arange(len(pep)) / self.LENGTH_SCALE
        )  # position info

        embedding[len(pep) + 1, 0] = 1  # padding info

        return embedding

    def convert_mgf_to_hdf5(self, hdf5_file, spec_type="hcd"):
        """
        Convert the loaded MGF data to HDF5 format

        Args:
            hdf5_file: Output HDF5 file path
            spec_type: Spectrum type (default: "hcd")
        """
        num_spectra = len(self.valid_spectra)
        print(f"Converting {num_spectra} valid spectra to HDF5 format")

        # Create HDF5 file
        print(f"Creating HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, 'w') as h5f:
            # Create datasets
            embedding_dset = h5f.create_dataset(
                'embedding',
                shape=(num_spectra, self.INPUT_LENGTH, self.INPUT_DIMENSION),
                dtype='float32'
            )
            meta_dset = h5f.create_dataset(
                'meta',
                shape=(num_spectra, *self.META_SHAPE),
                dtype='float32'
            )
            target_dset = h5f.create_dataset(
                'target',
                shape=(num_spectra, self.SPECTRA_DIMENSION),
                dtype='float32'
            )

            # Process and store each spectrum
            for idx in range(num_spectra):
                if idx % 1000 == 0:
                    print(f"Processing spectrum {idx}/{num_spectra}")

                spectrum = self.valid_spectra[idx]

                # preprocess single spectrum
                embedding = np.zeros((self.INPUT_LENGTH, self.INPUT_DIMENSION), dtype="float32")
                meta = np.zeros(self.META_SHAPE, dtype="float32")

                self.embed(spectrum, embedding=embedding)
                meta[0][spectrum["charge"] - 1] = 1  # charge
                meta[1][spectrum["type"]] = 1  # ftype
                meta[2][0] = self.safe_fastmass(spectrum["pep"], ion_type="M", charge=1) / self.PRECURSOR_SCALE

                if not "nce" in spectrum or spectrum["nce"] == 0:
                    meta[2][-1] = 0.25
                else:
                    meta[2][-1] = spectrum["nce"] / 100.0

                # compute y
                y = self.spectrum2vector(spectrum["mz"], spectrum["it"], spectrum["mass"], self.BIN_SIZE, spectrum["charge"])

                # Store in HDF5
                embedding_dset[idx] = embedding
                meta_dset[idx] = meta
                target_dset[idx] = y

        print(f"Successfully converted {num_spectra} spectra to {hdf5_file}")


    def vector2000_to_charge_records(self, vector_2000, vector_2000_ions, vector_2000_pos, max_charge=20):
        """
        Split 2000-dim vector into 100 records of 20 charge states each.
        Skip rows where all charge states are zero.
        """
        nseq = len(vector_2000) // (max_charge * 2)
        # reshape (100 records × 20 charges)
        reshaped = vector_2000.reshape(int(nseq), max_charge*2)
        v_b_records = reshaped[:, :max_charge]      
        v_y_records = reshaped[:, max_charge:]
        reshaped = np.vstack([v_b_records, v_y_records])        

        reshaped_ions = vector_2000_ions.reshape(int(nseq), max_charge * 2)
        v_b_records = reshaped_ions[:, :max_charge]      
        v_y_records = reshaped_ions[:, max_charge:]
        reshaped_ions = np.vstack([v_b_records, v_y_records])     

        reshaped_pos = vector_2000_pos.reshape(int(nseq), max_charge * 2)
        v_b_records = reshaped_pos[:, :max_charge]      
        v_y_records = reshaped_pos[:, max_charge:]
        reshaped_pos = np.vstack([v_b_records, v_y_records]) 

        # filter out all-zero rows
        mask = np.any(reshaped > 0, axis=1)
        valid_records = reshaped[mask]
        valid_ion_records = reshaped_ions[mask]
        valid_pos_records = reshaped_pos[mask]

        return valid_records, valid_ion_records, valid_pos_records
    

    def extract_ion_metadata(self, labels):
        ion_info = []
        for ion in labels:
            ion = ion.strip()
            if ion[0].lower() in ("b", "y"):
                ion_type = ion[0].lower()
                try:
                    position = int(ion[1:])
                except ValueError:
                    position = -1
                ion_info.append((ion_type, position))
        return ion_info
    
    def convert_mgf_to_hdf5_2000(self, hdf5_file, seq_length=50, max_charge=20):
        num_spectra = len(self.valid_spectra)
        print(f"Converting {num_spectra} valid spectra to HDF5 format")

        # Create HDF5 file
        print(f"Creating HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, 'w') as h5f:
            # Create datasets
            embedding_dset = h5f.create_dataset(
                'embedding',
                shape=(num_spectra, self.INPUT_LENGTH, self.INPUT_DIMENSION),
                dtype='float32'
            )
            meta_dset = h5f.create_dataset(
                'meta',
                shape=(num_spectra, *self.META_SHAPE),
                dtype='float32'
            )
            target_dset = h5f.create_dataset(
                'target',
                shape=(num_spectra, seq_length * max_charge *2), 
                dtype='float32'
            )

            # Process and store each spectrum
            for idx in range(num_spectra):
                if idx % 1000 == 0:
                    print(f"Processing spectrum {idx}/{num_spectra}")

                spectrum = self.valid_spectra[idx]

                # preprocess single spectrum
                embedding = np.zeros((self.INPUT_LENGTH, self.INPUT_DIMENSION), dtype="float32")
                meta = np.zeros(self.META_SHAPE, dtype="float32")

                self.embed(spectrum, embedding=embedding)
                meta[0][spectrum["charge"] - 1] = 1  # charge
                meta[1][spectrum["type"]] = 1  # ftype
                meta[2][0] = self.safe_fastmass(spectrum["pep"], ion_type="M", charge=1) / self.PRECURSOR_SCALE

                if not "nce" in spectrum or spectrum["nce"] == 0:
                    meta[2][-1] = 0.25
                else:
                    meta[2][-1] = spectrum["nce"] / 100.0

                # compute y
                y, ions, pos = self.spectrum2vector_2000(spectrum, seq_length=seq_length)
                if idx == 0:
                    l = len(y)
                    for i in range(l):
                        if y[i] != 0:
                            print(spectrum["pep"], i+1, "pos", (i//40 + 1), "charge", (i%40 + 1), y[i])

                # Store in HDF5
                embedding_dset[idx] = embedding
                meta_dset[idx] = meta
                target_dset[idx] = y

        print(f"Successfully converted {idx} spectra records to {hdf5_file}")


    def convert_mgf_to_hdf5_2000_simple(self, hdf5_file, seq_length=50, max_charge=20):
        num_spectra = len(self.valid_spectra)
        print(f"Converting {num_spectra} valid spectra to HDF5 format")
        # target_dim = (max_charge * 2) * seq_length  # = 2000
        nrecords = int((max_charge * 2 * seq_length) / max_charge)

        total_records = 0
        for spec in self.valid_spectra:
            y, ions, pos = self.spectrum2vector_2000(spec, seq_length=seq_length)
            inte, ion, p = self.vector2000_to_charge_records(y, ions, pos, max_charge=max_charge)
            total_records += len(inte)

        print(f"Total valide records {total_records}")
        # Create HDF5 file
        print(f"Creating HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, 'w') as h5f:
            # Create datasets
            embedding_dset = h5f.create_dataset(
                'embedding',
                shape=(total_records, self.INPUT_LENGTH, self.INPUT_DIMENSION),
                chunks=(1000, self.INPUT_LENGTH, self.INPUT_DIMENSION),
                dtype='float32', compression='gzip', compression_opts=4
            )
            meta_dset = h5f.create_dataset(
                'meta',
                shape=(total_records, *self.META_SHAPE),
                chunks=(1000, *self.META_SHAPE),
                dtype='float32', compression='gzip', compression_opts=4
            )
            target_dset = h5f.create_dataset(
                'target',
                shape=(total_records, self.SPECTRA_DIMENSION),
                chunks=(1000, self.SPECTRA_DIMENSION),
                dtype='float32', compression='gzip', compression_opts=4
            )

            # Process and store each spectrum
            idx2 = 0
            emb_batch_all = []
            meta_batch_all = []
            tgt_batch_all = []
            for idx in range(num_spectra):
                if idx % 1000 == 0:
                    print(f"Processing spectrum {idx}/{num_spectra}")

                spectrum = self.valid_spectra[idx]

                # preprocess single spectrum
                embedding = np.zeros((self.INPUT_LENGTH, self.INPUT_DIMENSION), dtype="float32")
                meta = np.zeros(self.META_SHAPE, dtype="float32")

                # embedding.fill(0)
                # meta.fill(0)

                self.embed(spectrum, embedding=embedding)
                meta[0][spectrum["charge"] - 1] = 1  # charge
                meta[1][spectrum["type"]] = 1  # ftype
                meta[2][0] = self.safe_fastmass(spectrum["pep"], ion_type="M", charge=1) / self.PRECURSOR_SCALE

                if not "nce" in spectrum or spectrum["nce"] == 0:
                    meta[2][-1] = 0.25
                else:
                    meta[2][-1] = spectrum["nce"] / 100.0

                # compute y
                y,ions, pos = self.spectrum2vector_2000(spectrum, seq_length=seq_length) 
                # convert to new records
                inte_records, ions_records, pos_records = self.vector2000_to_charge_records(y, ions, pos, max_charge=max_charge) 
                # print(f"the dimension: {np.shape(inte_records)}")
                # Store in HDF5
                # ion_info = self.extract_ion_metadata(spectrum["labels"])
                emb_batch = []
                meta_batch = []
                tgt_batch = []
                for chunk, i_chunk, p_chunk in zip(inte_records, ions_records, pos_records):
                    if idx2 % 5000 == 0:
                        print(f"Processing record {idx2}/{total_records}")
                    meta_chunk = np.copy(meta)
                    meta_chunk[3][:] = 0
                    meta_chunk[4][:] = 0

                    if np.any(i_chunk == 1):
                        meta_chunk[3][0] = 1
                    if np.any(i_chunk == 2):
                        meta_chunk[3][1] = 1

                    nonzero_pos = p_chunk[p_chunk > 0]
                    meta_chunk[4][nonzero_pos - 1] = 1
                
                    emb_batch.append(embedding)
                    meta_batch.append(meta_chunk)
                    tgt_batch.append(chunk)

                    # embedding_dset[idx2] = embedding
                    # meta_dset[idx2] = meta_chunk
                    # target_dset[idx2] = chunk
                    # idx2 += 1

                # n_batch = len(emb_batch)
                # embedding_dset[idx2:idx2 + n_batch] = np.stack(emb_batch)
                # meta_dset[idx2:idx2 + n_batch] = np.stack(meta_batch)
                # target_dset[idx2:idx2 + n_batch] = np.stack(tgt_batch)
                # idx2 += n_batch

                emb_batch_all.extend(emb_batch)
                meta_batch_all.extend(meta_batch)
                tgt_batch_all.extend(tgt_batch)
                
                if len(emb_batch_all) >= 5000 or idx == num_spectra - 1:
                    start = idx2
                    end = idx2 + len(emb_batch_all)
                    embedding_dset[start:end] = np.stack(emb_batch_all)
                    meta_dset[start:end] = np.stack(meta_batch_all)
                    target_dset[start:end] = np.stack(tgt_batch_all)
                    idx2 = end
                    emb_batch_all.clear()
                    meta_batch_all.clear()
                    tgt_batch_all.clear()

                    print(f"Written up to {idx}/{num_spectra}")

        print(f"Successfully converted {idx2} spectra records to {hdf5_file}")
