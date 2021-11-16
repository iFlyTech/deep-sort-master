"""
Parses commmand line arguments for generate.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="the name of the dataset (e.g. train, val, test)")
parser.add_argument("--size", type=int, help="size of the dataset to generate")
parser.add_argument("--max_val", type=int, help="the maximum value in an array")
parser.add_