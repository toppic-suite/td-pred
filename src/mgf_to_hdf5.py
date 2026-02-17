#!/usr/bin/env python3

import numpy as np
import argparse
import mgf_torch_generator


def main():
    parser = argparse.ArgumentParser(description="Convert MGF file to HDF5 format")
    parser.add_argument(
        "--mgf", type=str, required=True, help="Input MGF file path"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output HDF5 file path"
    )
    parser.add_argument(
        "--type", type=str, default="hcd",
        choices=["unknown", "cid", "etd", "hcd", "ethcd", "etcid"],
        help="Spectrum type (default: hcd)"
    )
    parser.add_argument("--seq_length", type=int, default=200)
    parser.add_argument("--output_type", type=str, default="anno",
                        choices=["bin", "anno"], help="Output type (default: anno)")
    args = parser.parse_args()
    print("sequence length", args.seq_length)
    if args.output_type == "bin":
        converter = mgf_torch_generator.MgfBatchGenerator(args.mgf)
        converter.convert_mgf_to_hdf5(args.out, spec_type=args.type)
    else:
        dataset = mgf_torch_generator.MgfBatchGenerator(args.mgf, max_peptide_length=args.seq_length)
        print("Converting spectra to HDF5...")
        dataset.convert_mgf_to_hdf5_2000(args.out, seq_length=args.seq_length)
        print("Done! Output file saved at:", args.out)

if __name__ == "__main__":
    main()
