#!/usr/bin/env python3
import argparse
import torch
import model_data as md
import td_pred_model
from model_data import get_mono_mass_list


def read_sequences(input_file):
    seqs = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line: #and not line.startswith(">"):
                seqs.append(line)
    return seqs


def predict_intensity(model, sequence, precursor_charge, nce=25,
                      activation="hcd", instrument="q exactive",
                      max_len=200, device="cpu"):

    spectrum = {
        "proteoform": sequence,
        "prec_charge": precursor_charge,
        "activation_type": activation,
        "instrument": instrument,
        "nce": nce,
    }

    seq_enc = md.encode_spectrum(spectrum, max_len)
    meta = md.encode_meta(spectrum)
    mask = md.get_mask(spectrum, max_len)
    charge_mask = md.get_charge_mask(precursor_charge)

    # add batch dimension
    seq_enc = torch.tensor(seq_enc).unsqueeze(0).to(device)
    meta = torch.tensor(meta).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    charge_mask = torch.tensor(charge_mask).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(seq_enc, meta, mask, charge_mask)

    return pred.cpu().numpy().reshape(max_len-1, -1)


PROTON = 1.007276466

def compute_b_y_masses(sequence):
    mono = get_mono_mass_list()
    prefix = []
    suffix = []

    cur = 0
    for aa in sequence[:-1]:
        cur += mono[aa]
        prefix.append(cur + PROTON)   # b-ion base

    cur = 0
    for aa in reversed(sequence[1:]):
        cur += mono[aa]
        suffix.append(cur + 18.01056 + PROTON)  # y-ion base

    return prefix, suffix


def build_predicted_spectrum(sequence, pred_tensor, max_charge=30):
    b_mass, y_mass = compute_b_y_masses(sequence)
    peaks = []

    L = len(sequence) - 1

    for i in range(L):
        for z in range(1, max_charge+1):
            b_int = pred_tensor[i, z-1]
            y_int = pred_tensor[i, max_charge + z-1]

            if b_int > 0.01:
                mz = (b_mass[i] + z*PROTON) / z
                peaks.append((mz, b_int, z, "b", i+1))

            if y_int > 0.01:
                mz = (y_mass[i] + z*PROTON) / z
                peaks.append((mz, y_int, z, "y", i+1))

    return sorted(peaks, key=lambda x: x[0])


def write_msalign(peaks, sequence, outfile,
                  precursor_charge, activation, instrument, nce):

    with open(outfile, "a") as f:   
        f.write("BEGIN IONS\n")
        f.write(f"PRECURSOR_CHARGE={precursor_charge}\n")
        f.write(f"ACTIVATION={activation}\n")
        f.write(f"INSTRUMENT={instrument}\n")
        f.write(f"COLLISION_ENERGY={nce}\n")
        f.write(f"DATABASE_SEQUENCE={sequence}\n")

        for mz, inten, ch, itype, pos in peaks:
            f.write(f"{mz:.5f}\t{inten:.4f}\t{ch}\t0\t{itype}{pos}\t{pos}\n")

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
    parser.add_argument("--input", required=True, help="file with sequences")
    parser.add_argument("--output", required=True, help="output msalign")
    parser.add_argument("--charge", type=int, default=4)
    parser.add_argument("--nce", type=int, default=25)
    parser.add_argument("--activation", default="hcd")
    parser.add_argument("--instrument", default="q exactive")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    # load model
    model = td_pred_model.TransformerSeq2Seq(...)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # clear output file
    open(args.output, "w").close()

    sequences = read_sequences(args.input)

    for i, seq in enumerate(sequences):
        print(f"Predicting spectrum {i+1}/{len(sequences)}: {seq}")

        pred = predict_intensity(
            model,
            seq,
            args.charge,
            nce=args.nce,
            activation=args.activation,
            instrument=args.instrument
        )

        peaks = build_predicted_spectrum(seq, pred)

        write_msalign(
            peaks,
            seq,
            args.output,
            args.charge,
            args.activation,
            args.instrument,
            args.nce
        )
