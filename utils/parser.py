import sys
import argparse

import torch


def parse_args_sparsity():

    parser = argparse.ArgumentParser(description='Argument parser for test_sparsity.py')

    parser.add_argument('--size', default=1000, type=int,
                        help='size of the matrix to be multiplied')

    parser.add_argument('--iterations', default=10, type=int,
                        help='number of iterations for each sparsity level')

    parser.add_argument('--sparsities', default=[.95, .98, .99, .995, .997, .999], type=float,
                        nargs="*", help='the levels of sparsities to test')

    parser.add_argument('--verbose', default=False, type=bool,
                        help='print the progress of the sparsity_test')

    parser.add_argument('--show_plot', default=True, type=bool,
                        help='show the plot of the sparsity_test')

    return parser.parse_args()
