import numpy as np
import os


class ResDataSaver:
    def __init__(self, res_matrix=None, weights=None, factor_params=None, filenames=None, file_dir=None):
        self.res_matrix = res_matrix
        self.filenames = filenames
        self.file_dir = file_dir
        self.weights = weights
        self.factor_params = factor_params

    def save_res_matrix(self, res_matrix, filename, file_dir=None):
        save_file_path = ''
        if file_dir is not None and len(filename) is 1:
            save_file_path = os.path.normpath(os.path.join(file_dir, filename[0]))
        elif len(filename) is 1:
            save_file_path = os.path.normpath(filename[0])
        np.save(save_file_path, res_matrix)

    def save_res_matrix_from_weights(self, weights, factor_params, filenames, file_dir=None):
        save_file_path = ''
        save_file_path_2 = ''
        if file_dir is not None and len(filenames) is 2:
            save_file_path = os.path.normpath(os.path.join(file_dir, filenames[0]))
            save_file_path_2 = os.path.normpath(os.path.join(file_dir, filenames[1]))
        elif len(filenames) is 2:
            save_file_path = os.path.normpath(filenames[0])
            save_file_path_2 = os.path.normpath(filenames[1])
        np.save(save_file_path, weights)
        np.save(save_file_path_2, factor_params)

    def __call__(self, *args, **kwargs):
        if self.weights is not None and self.factor_params is not None:
            self.save_res_matrix_from_weights(self.weights, self.factor_params, self.filenames, file_dir=self.file_dir)
        elif self.weights is None and self.factor_params is None:
            self.save_res_matrix(self.res_matrix, self.filenames, file_dir=self.file_dir)
