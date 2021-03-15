#!/usr/bin/env python3

import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


def verify_str_arg(value, valid_values):
    assert value in valid_values
    return value


def image_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def load_labels(path):
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [
            {
                headers[column_index]: row[column_index]
                for column_index in range(len(row))
            }
            for row in reader
        ]
