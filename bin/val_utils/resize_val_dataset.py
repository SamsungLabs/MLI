import argparse
import os
import logging
from copy import copy
from timeit import default_timer as timer
import json
imort

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


def main():
    logger = logging.getLogger('root')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to the config file.')
    parser.add_argument('--output-path', type=str, help='Path to outputs directory.')
    parser.add_argument('--height', type=int, help='Width of final images.')
    parser.add_argument('--width', type=int, help='Width of final images.')

    opts = parser.parse_args()

    base_path = opts.dataset_path
    # TODO script for resizing dataset

