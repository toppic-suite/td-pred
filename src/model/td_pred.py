#!/usr/bin/env python3
import argparse
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F

import model_data as md
import td_pred_model
from model_data import get_mono_mass_list


def predict_intensity(model, spectrum, max_len, device):
    seq_enc = md.encode_spectrum(spectrum, max_len)
    meta = md.encode_meta(spectrum)
    mask = md.get_mask(spectrum, max_len)
    precursor_charge = spectrum["prec_charge"]
    charge_mask = md.get_charge_mask(precursor_charge)

    # add batch dimension
    seq_enc = torch.tensor(seq_enc).unsqueeze(0).to(device)
    meta = torch.tensor(meta).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    charge_mask = torch.tensor(charge_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(seq_enc, meta, mask, charge_mask)

    return pred.cpu().numpy().reshape(max_len-1, -1)


def compute_b_y_masses(sequence):
    mono = get_mono_mass_list()
    prefix = []
    suffix = []

    cur = 0
    for aa in sequence[:-1]:
        cur += mono[aa]
        prefix.append(cur)   # b-ion base

    cur = 0
    for aa in reversed(sequence[1:]):
        cur += mono[aa]
        suffix.append(cur + 18.01056)  # y-ion base

    return prefix, suffix


def build_predicted_spectrum(sequence, activation, pred_tensor, max_charge=30):
    b_mass, y_mass = compute_b_y_masses(sequence)
    peaks = []

    seq_len = len(sequence)
    pep_bond_num = seq_len - 1
    n_shift = 0
    c_shift = 0
    n_ion = "b"
    c_ion = "y"
    if activation == "etd":
        n_shift = 0
        c_shift = 0 
        n_ion = "c"
        c_ion = "z_dot"
    for i in range(pep_bond_num):
        for z in range(1, max_charge+1):
            n_int = pred_tensor[i, z-1]
            if n_int > 0.01:
                n_mass = b_mass[i] + n_shift
                peaks.append((n_mass, n_int, z, n_ion, i+1))
            c_int = pred_tensor[i, max_charge + z-1]
            if c_int > 0.01:
                c_mass = y_mass[i] + c_shift
                peaks.append((c_mass, c_int, z, c_ion, seq_len - (i+1)))

    return sorted(peaks, key=lambda x: x[0])


def write_msalign(peaks, spectrum, outfile):
    precursor_charge = spectrum["prec_charge"]
    activation = spectrum["activation_type"]
    instrument = spectrum["instrument"]
    nce = spectrum["nce"]
    with open(outfile, "a") as f:   
        f.write("BEGIN IONS\n")
        f.write(f"PRECURSOR_CHARGE={precursor_charge}\n")
        f.write(f"ACTIVATION={activation}\n")
        f.write(f"INSTRUMENT={instrument}\n")
        f.write(f"COLLISION_ENERGY={nce}\n")
        f.write(f"DATABASE_SEQUENCE={spectrum['proteoform']}\n")

        for mass, inten, ch, itype, pos in peaks:
            f.write(f"{mass:.5f}\t{inten:.4f}\t{ch}\t0\t{itype}{pos}\t{pos}\n")

        f.write("END IONS\n\n")


def predict_msalign_from_sequence(model, sequence,
                                  precursor_charge=4,
                                  activation="hcd",
                                  instrument="q exactive",
                                  nce=25,
                                  outfile="pred.msalign",
                                  device="cpu"):

    pred = predict_intensity(
        model, sequence, precursor_charge,
        nce=nce, activation=activation,
        instrument=instrument,
        device=device
    )

    peaks = build_predicted_spectrum(sequence, pred)
    write_msalign(peaks, sequence, outfile,
                  precursor_charge, activation, instrument, nce)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="tsv file with sequence and meta information")
    parser.add_argument("--model", required=True, help="model file")
    parser.add_argument("--output", required=True, help="output msalign file")
    args = parser.parse_args()


   # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_len = 199  
    output_dim = 60
    max_seq_length = 200
    single_model = td_pred_model.TransformerSeq2Seq(
        output_len=output_len,
        output_dim=output_dim,
        max_seq_length=max_seq_length,
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(single_model)
    else:
        model = single_model

    # Load presaved model if specified
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")

    model.eval()

    # clear output file
    open(args.output, "w").close()

    df = pandas.read_csv(args.input, sep="\t")

    for i in range(len(df)):
        print(f"Predicting spectrum {i+1}/{len(df)}: {seq}")
        seq = df.loc[i, "DATABASE_SEQUENCE"]
        charge = df.loc[i, "PRECURSOR_CHARGE"]
        activation = df.loc[i, "ACTIVATION"]
        instrument = df.loc[i, "INSTRUMENT"]
        nce = df.loc[i, "COLLISION_ENERGY"]

        spectrum = {
            "proteoform": seq,
            "prec_charge": charge,
            "activation_type": activation,
            "instrument": instrument,
            "nce": nce,
        }

        pred = predict_intensity(model, spectrum, max_seq_length, device)
        peaks = build_predicted_spectrum(seq, pred)

        write_msalign(peaks, spectrum, args.output)
