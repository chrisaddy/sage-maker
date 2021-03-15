#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys
from os.path import join, realpath, dirname
import urllib.request
import shutil
import ssl
import io
from PIL import Image
import numpy as np


# ssl._create_default_https_context = ssl._create_unverified_context  # dirty fix
script_dir = dirname(realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the KaoKore Dataset.")
    parser.add_argument(
        "--force",
        help="Force redownloading of already downloaded images",
        action="store_true",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of simultaneous threads to use for downloading",
        default=16,
    )
    parser.add_argument(
        "--ssl_unverified_context",
        help="Force to use unverified context for SSL",
        action="store_true",
    )
    args = parser.parse_args()

    if args.ssl_unverified_context:
        print(
            "[WARN] Use unverified context for SSL as requested. Use at your own risk"
        )
        ssl._create_default_https_context = ssl._create_unverified_context

    script_dir = dirname(realpath(__file__))
    dataset_suffix = {"1.0": "", "1.1": "_v1.1", "1.2": "_v1.2"}[args.dataset_version]
    script_dataset_dir = join(script_dir, "dataset{}".format(dataset_suffix))

    urls_file = join(script_dataset_dir, "urls.txt")
    iurls = load_urls(urls_file)

    redownloading_warning = False

    print(
        "Downloading Kaokore version {}, saving to {}".format(
            args.dataset_version, args.dir
        )
    )
    print("Downloading {} images using {} threads".format(len(iurls), args.threads))
    images_dir = join(args.dir, "images_256")
    os.makedirs(images_dir, exist_ok=True)

    if tqdm:  # Use tqdm progressbar
        bar = tqdm(total=len(iurls))
        for i, _ in enumerate(pool.imap_unordered(download_and_check_image, iurls)):
            bar.update()
    print()

    # TODO: Download these files if not already present
    for file in [
        "labels.csv",
        "original_tags.txt",
        "labels.metadata.en.txt",
        "labels.metadata.ja.txt",
    ]:
        shutil.copy(join(script_dataset_dir, file), args.dir)
