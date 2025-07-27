#!/usr/bin/env python
"""Command line entry for preprocessing CheXpert images."""
import argparse
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main.preprocessing.chexpert_preprocess import process_directory


def main():
    parser = argparse.ArgumentParser(description="Preprocess CheXpert dataset")
    parser.add_argument("src", type=str, help="Input directory of raw images")
    parser.add_argument("dst", type=str, help="Destination directory")
    parser.add_argument("--size", type=int, default=224, help="Target square size")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE enhancement")
    args = parser.parse_args()

    process_directory(args.src, args.dst, size=args.size, apply_clahe=not args.no_clahe)


if __name__ == "__main__":
    main()
