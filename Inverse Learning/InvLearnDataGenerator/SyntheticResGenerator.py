import sys
import numpy as np
import argparse
import math
import scipy.sparse
from InverseLearning.InvLearnDataGenerator.ResDataSaver import ResDataSaver


class SyntheticResGenerator:
    def __init__(self, res_shape=None, res_range=None, num_of_weights=None, weights_range=None, no_data_value=-9999,
                 no_data_indices=None, filenames=None, file_dir=None, seed=42):
        self.res_shape = res_shape
        self.res_range = res_range
        self.no_data_value = no_data_value
        self.seed = seed
        self.num_of_weights = num_of_weights
        self.weights_range = weights_range
        self.no_data_indices = no_data_indices
        self.filenames = filenames
        self.file_dir = file_dir

    def generate_res_matrix(self, res_shape, res_range, no_data_indices, no_data_value=-9999, seed=42):
        np.random.seed(seed)
        res_matrix = np.random.random(res_shape)
        res_matrix = res_matrix*res_range
        np.fill_diagonal(res_matrix, 0.0)
        if no_data_indices is not None:
            res_matrix[no_data_indices] = no_data_value
        return res_matrix

    def generate_res_matrix_from_weights(self, res_shape, res_range, num_of_weights, weights_range, no_data_indices,
                                         no_data_value=-9999, seed=42):
        np.random.seed(seed)
        weights = [np.random.randint(weights_range, size=(1, 1)) for _ in range(num_of_weights)]
        true_res_range = math.ceil(res_range/(max(weights)[0][0]*num_of_weights))
        print('True res range: ', true_res_range)
        #self.create_base_matrices(16)
        #self.create_base_matrices(16)
        #factor_params = [scipy.sparse.rand(res_shape[0], res_shape[1], density=0.2, format='csr').
                             #todense()*true_res_range for _ in range(num_of_weights)]
        #print('True res range: ', true_res_range)
        factor_params = [self.create_base_matrices(res_shape[0])*true_res_range for _ in range(num_of_weights)]
        #factor_params = [np.random.randint(true_res_range, size=res_shape) for _ in range(num_of_weights)]
        factor_params = [self.fill_diagonal(factor) for factor in factor_params]
        #print(factor_params)
        #print(weights)
        res_matrix = sum([x*y for x, y in zip(weights, factor_params)])
        if no_data_indices is not None:
            res_matrix[no_data_indices] = no_data_value
        return tuple((res_matrix, weights, factor_params))

    '''
    n = must be a perfect square
    '''
    def create_base_matrices(self, n):
        m = int(np.sqrt(n))
        # Make sure that the np.random.rand creates a different random matrix each call
        asc_arr = np.random.rand(m, m)
        #print(asc_arr)
        base_matrix = np.zeros((n, n))
        c = 0
        for i in range(m):
            for j in range(m):
                self.populate_res_matrix_cells(base_matrix, asc_arr, i, j, c, m)
                c = c + 1

        #print('Base matrices')
        #print(asc_arr)
        #print(base_matrix)
        return base_matrix


    def populate_res_matrix_cells(self, res_matrix, matrix, i, j, col, m_dim):
        # Top
        if (i-1) >= 0:
            res_matrix[col-m_dim, col] = (matrix[i, j] + matrix[i-1, j])/2.0
        # Bottom
        if (i+1) < m_dim:
            res_matrix[col+m_dim, col] = (matrix[i, j] + matrix[i+1, j])/2.0
        # Left
        if (j-1) >= 0:
            res_matrix[col-1, col] = (matrix[i, j] + matrix[i, j-1])/2.0
        # Right
        if (j+1) < m_dim:
            res_matrix[col+1, col] = (matrix[i, j] + matrix[i, j+1])/2.0

    def fill_diagonal(self, matrix):
        matrix = np.asarray(matrix)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def __call__(self, *args, **kwargs):
        if self.num_of_weights is not None and self.weights_range is not None:
            data = self.generate_res_matrix_from_weights(self.res_shape, self.res_range, self.num_of_weights,
                                                         self.weights_range, self.no_data_indices, self.no_data_value,
                                                         self.seed)
            if self.filenames is not None:
                rds = ResDataSaver(weights=data[1], factor_params=data[2], filenames=self.filenames,
                                   file_dir=self.file_dir)
                rds()
            print(data[1])
            return data
        else:
            data = self.generate_res_matrix(self.res_shape, self.res_range, self.no_data_indices,
                                            self.no_data_value, self.seed)
            if self.filenames is not None:
                rds = ResDataSaver(res_matrix=data, filenames=self.filenames, file_dir=self.file_dir)
                rds()
            print(data[1])
            return data


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Generate resistance matrix directly or compute from randomly generated'
                                                 ' factor parameters.')
    parser.add_argument("res_shape", help="Desired shape of generated resistance matrix. Type the same value twice.",
                        nargs=2, type=int)
    parser.add_argument("res_range", help="Desired range of values in generated resistance matrix.", type=int)
    parser.add_argument('--num_weights', help="Size of weight vector in resistance factor parameter calculation. Only "
                                              "specify argument if using factor parameters", dest='n_w', default=None,
                        const=None, nargs='?', action='store', type=int)
    parser.add_argument('--w_range', help="Desired range of values weight vector can take. Only specify argument if "
                                          "using factor parameters.", dest='w_r', default=100, const=100, nargs='?',
                        action='store', type=int)
    parser.add_argument('--no_data_indices', help='Binary .npy file storing which indices on resistance matrix should '
                                                  'take on no data value. Can be relative or full file path.',
                        dest='ndi', default=None, const=None, nargs=1)
    parser.add_argument('--no_data_value', help="Specifies no data value.", dest="ndv", default=-9999, const=-9999,
                        nargs='?', type=int)
    parser.add_argument('--filenames', help="List of filenames to store resistance matrix data. List one filename for "
                                            "storing resistance matrix or two filenames for storing weights and factor "
                                            "parameters.", dest="file", default=None, const=None, nargs='*')
    parser.add_argument('--file_dir', help="File directory to store resistance matrix data. If not specified, "
                                           "--filenames argument must contain relative or absolute file paths.",
                        dest="dir", default=None, const=None, nargs='?')
    parser.add_argument("--seed", help="Seed of random number generator, automatically defaults to 42.", action='store',
                        default=42, dest='seed', nargs='?', const=42, type=int)
    args = parser.parse_args()
    srg = SyntheticResGenerator(res_shape=tuple(args.res_shape), res_range=args.res_range, num_of_weights=args.n_w,
                                weights_range=args.w_r, no_data_value=args.ndv, no_data_indices=args.ndi,
                                seed=args.seed, filenames=args.file, file_dir=args.dir)
    print(srg())
    #srg()
