import sys
import numpy as np
import argparse
import os
import pandas as pd


class ResDataReader:
    def __init__(self, res_file=None, file_dir=None):
        self.res_file = res_file
        self.file_dir = file_dir

    def read_res_file(self, res_file, file_dir=None):
        # Reading in ESRI ASCII grid format as pandas dataframes
        res_file_path = ''
        if file_dir is not None:
            res_file_path = os.path.normpath(os.path.join(file_dir, res_file))
        else:
            res_file_path = os.path.normpath(res_file)
        header = pd.read_csv(res_file_path, delim_whitespace=True, nrows=6, header=None)
        matrix_arr = pd.read_csv(res_file_path, delim_whitespace=True, header=None, skiprows=6)

        # Convert dataframe into numpy ndarray
        matrix = matrix_arr.values

        '''
        # Populate resistance matrix based on grid resistance given by .asc file
        N = np.size(matrix_arr)
        res_matrix = np.zeros((N, N))
        no_data_val = header.iloc[5, 1]
        for col in range(N):
            i = col // matrix_arr.shape[0]
            j = col % matrix_arr.shape[1]
            if matrix[i, j] != no_data_val:
                ResDataReader.populate_res_matrix_cells(res_matrix, matrix, i, j, col, matrix.shape[1], no_data_val)
        '''
        return matrix

    @staticmethod
    def populate_res_matrix_cells(res_matrix, matrix, i, j, col, m_dim, no_data_val):
        # Top
        if (i-1) >= 0 and matrix[i-1, j] != no_data_val:
            res_matrix[col-m_dim, col] = (matrix[i, j] + matrix[i-1, j])/2.0
        # Bottom
        if (i+1) < m_dim and matrix[i+1, j] != no_data_val:
            res_matrix[col+m_dim, col] = (matrix[i, j] + matrix[i+1, j])/2.0
        # Left
        if (j-1) >= 0 and matrix[i, j-1] != no_data_val:
            res_matrix[col-1, col] = (matrix[i, j] + matrix[i, j-1])/2.0
        # Right
        if (j+1) < m_dim and matrix[i, j+1] != no_data_val:
            res_matrix[col+1, col] = (matrix[i, j] + matrix[i, j+1])/2.0

    def __call__(self, *args, **kwargs):
        return self.read_res_file(self.res_file, self.file_dir)

'''
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Read in resistance .asc file in ESRI ASCII grid format.')
    parser.add_argument("filename", help=".asc file to extract resistance matrix, must contain absolute path if file "
                                         "directory is not specified")
    parser.add_argument("directory", help="File directory where .asc file is located. Optional to indicate.", nargs='?')
    args = parser.parse_args()
    rd = ResDataReader(args.filename, args.directory)
    print(rd())
'''

