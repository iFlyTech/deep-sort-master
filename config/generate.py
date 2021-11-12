"""
Parses commmand line arguments for generate.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="the name of the dataset (e.g. t