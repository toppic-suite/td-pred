#!/usr/bin/env python3

import argparse
import msalign_anno_torch_generator


def main():
    parser = argparse.ArgumentParser(description="Convert MSALIGN anno file to HDF5 format")
    parser.add_argument(
        "--msalign", type=str, required=True, help="Input MSALIGN file path"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output HDF5 file path"
    )
    parser.add_argument("--max_length", type=int, default=200)
    args = parser.parse_args()
    print("sequence length", args.max_length)

    dataset = msalign_anno_torch_generator.MsalignAnnoBatchGenerator(args.msalign, max_seq_length=args.max_length)
    print("Converting spectra to HDF5...")
    dataset.convert_anno_msalign_to_hdf5(args.out)
    print("Done! Output file saved at:", args.out)

if __name__ == "__main__":
    main()
