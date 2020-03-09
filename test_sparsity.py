import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from utils.parser import parse_args_sparsity


def make_matrix(size, sparsity):
    return sp.random(size, size, density=1-sparsity)


def main():

    size = args.size
    n_iter = args.iterations
    sparsities = args.sparsities
    sparsities.sort()
    verbose = args.verbose
    
    args = parse_args_sparsity()
    print('------ Parameters for test_sparsity ------')
    for parameter, value in args.__dict__.items():
        print(f'{parameter}: {value}')
    print('------------------------------------------')

    dense = []
    sparses = []
    for s in sparsities:
        if verbose:
            print(f'------ Testing {s} sparsity ------')

        dur_dense = 0
        for _ in range(n_iter):
            A = make_matrix(size, s).toarray()
            B = make_matrix(size, s).toarray()
            
            t = time.time()
            A @ B  # multiplication with numpy matrices
            dur_dense += time.time() - t
        
        if verbose:
            print(f'Mean of {n_iter} dense multiplications:  {dur_dense/n_iter:.4f} s')
        
        dense.append(dur_dense)


        dur_sparse = 0
        for _ in range(n_iter):
            A = make_matrix(size, s)
            B = make_matrix(size, s)
            
            t = time.time()
            A.multiply(B)  # multiplication with scipy sparse matrices
            dur_sparse += time.time() - t
                
        if verbose:
            print(f'Mean of {n_iter} sparse multiplications: {dur_sparse/n_iter:.4f} s')
        
        sparses.append(dur_sparse)

    sparsities_percent = [100 * s for s in sparsities]

    if args.show_plot:
        plt.plot(sparsities_percent, dense, label='Dense')
        plt.plot(sparsities_percent, sparses, label='Sparse')
        plt.legend()
        plt.xlabel('Sparsity (%)')
        plt.ylabel(f'Time for {n_iter} multiplications (s)')
        plt.show()

    return None


if __name__ == '__main__':
    main()
