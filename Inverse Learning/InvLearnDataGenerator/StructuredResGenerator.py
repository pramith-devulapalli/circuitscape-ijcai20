import sys
import numpy as np
import argparse
import math
import copy
from InverseLearning.InvLearnDataGenerator.ResDataSaver import ResDataSaver


class StructuredResGenerator:
    def __init__(self, res_grid_shape=None, num_of_weights=None, angle=None, sigma_x=None, sigma_y=None, mean_x=None,
                 mean_y=None, filenames=None, file_dir=None, seed=42, starting_point=None, ending_point=None):
        self.res_grid_shape = res_grid_shape
        self.num_of_weights = num_of_weights
        self.starting_point = starting_point
        self.ending_point = ending_point
        self.angle = angle
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.filenames = filenames
        self.file_dir = file_dir
        self.seed = seed

    '''
    Generate the resistance grid from a uniform distribution and specified grid shape
    '''
    def generate_resistance_grid(self, res_grid_shape):
        res_grid = np.random.uniform(low=0.8, high=1.0, size=res_grid_shape)
        return res_grid

    '''
    Generate a set of points such that they form an elliptical set of coordinates within the resistance grid
    '''
    def generate_elliptical_mapping(self, angle, sigma_x, sigma_y, mean_x, mean_y, seed=42):
        np.random.seed(seed)
        mean = [0, 0]
        cov = np.array([[sigma_x, 0], [0, sigma_y]])
        x, y = np.random.multivariate_normal(mean, cov, 100).T
        x_new, y_new = self.rotate_ellipse(x, y, angle)
        x_new = x_new + mean_x
        y_new = y_new + mean_y
        return x_new, y_new

    def rotate_ellipse(self, x, y, angle):
        coordinates = np.column_stack((x, y)).T
        rotation_matrix = np.array([[np.cos(angle*np.pi/180), -1.0*np.sin(angle*np.pi/180)], [np.sin(angle*np.pi/180),
                                                                                              np.cos(angle*np.pi/180)]])
        new_coord = np.matmul(rotation_matrix, coordinates)
        return new_coord

    # Old function
    '''
    def parameters(self, angle, sig_x, sig_y):
        a = np.cos(angle*np.pi/180)**2/(2*sig_x**2) + np.sin(angle*np.pi/180)**2/(2*sig_y**2)
        b = -1.0*(np.sin(2.0*angle*np.pi/180)/(4*sig_x**2)) + (np.sin(2*angle*np.pi/180)/(4*sig_x**2))
        c = np.sin(angle*np.pi/180)**2/(2*sig_x**2) + np.cos(angle*np.pi/180)**2/(2*sig_y**2)
        cov = np.array([[a, b], [b, c]])
        return cov
    '''

    '''
    Create the structured resistance grid from the elliptical mapping
    '''
    def structured_resistance_grid(self, res_grid, x_elliptical, y_elliptical, start, end):
        # Prune in the x-direction
        pairs = np.column_stack((x_elliptical, y_elliptical)).T
        #print(pairs)
        #print(start[0])
        #print(end[0])
        pairs_y = pairs[:, (pairs[0] >= start[0]) & (pairs[0] <= end[0])]
        #print('Pruned in the x-direction')
        #print(pairs_y)

        # Prune in the y-direction
        pairs_pruned = pairs_y[:, (pairs_y[1] >= start[1]) & (pairs_y[1] <= end[1])]
        #print('Pruned in the y-direction')
        #print(pairs_pruned)
        ind = pairs_pruned.astype('int')
        #print('Converted to int')
        #print(ind)

        # Place ellipse into res_grid
        # The 0th column
        #print(ind[0])
        # The 1st column
        #print(ind[1])
        num_elements = np.size(res_grid[ind[0], ind[1]])
        #print(res_grid[ind[0], ind[1]])
        #print('Number of elements:')
        #print(num_elements)
        # Build a for loop
        res_grid[ind[0], ind[1]] = np.random.uniform(low=0.01, high=0.1, size=num_elements)
        return res_grid

    '''
    Save the resistance grid
    '''
    def save_resistance_grid(self, res_grid, filename, file_dir=None):
        print(filename[0])
        rds = ResDataSaver(res_matrix=res_grid, filenames=[filename[0]],  file_dir=file_dir)
        rds()

    '''
    Convert the resistance grid to a resistance matrix using the helper function populate_res_matrix_cells
    '''
    def resistance_grid_to_res_matrix(self, res_grid, res_grid_shape):
        n = res_grid_shape[0]*res_grid_shape[1]
        res_matrix = np.zeros((n, n))
        c = 0
        for i in range(res_grid_shape[0]):
            for j in range(res_grid_shape[1]):
                self.populate_res_matrix_cells(res_matrix, res_grid,  i, j, c, res_grid_shape[0])
                c = c + 1
        res_matrix = self.fill_diagonal(res_matrix)
        return res_matrix

    def fill_diagonal(self, matrix):
        matrix = np.asarray(matrix)
        np.fill_diagonal(matrix, 0.0)
        return matrix

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

    '''
    Convert the resistance matrix to a conductance matrix
    '''
    def res_matrix_to_cond_matrix(self, res_matrix):
        res_matrix[res_matrix == 0] = float('Inf')
        return np.reciprocal(res_matrix)

    '''
    Create the base matrices and the true weight vector according to the conductance matrix
    '''
    def generate_cond_parametrization(self, cond_matrix, num_of_weights):
        # Maximum value in cond_matrix
        max_val = np.max(cond_matrix)
        #base_matrices_list = [np.random.uniform(low=0.01, high=max_val/(num_of_weights+1), size=cond_matrix.shape)
                              #for _ in range(num_of_weights - 1)]
        base_matrices_list = []
        weights = [np.random.randint(low=10, high=70, size=(1, 1)) for _ in range(num_of_weights)]
        temp_matrix = copy.deepcopy(cond_matrix)
        for i in range(num_of_weights):
            if i == (num_of_weights - 1):
                base_matrices_list.append(temp_matrix/weights[i])
            else:
                base_matrix = np.random.uniform(low=0.01, high=max_val/(num_of_weights+1), size=cond_matrix.shape)
                base_matrix = self.fill_diagonal(base_matrix)
                c = temp_matrix - base_matrix
                ind = c < 0
                base_matrix[ind] = temp_matrix[ind]
                base_matrices_list.append(base_matrix/weights[i])
                temp_matrix = temp_matrix - base_matrix

        return base_matrices_list, weights

    def compute_res_matrix(self, weights, res_factors):
        #print(weights)
        #print(res_factors)
        r = sum([x*y for x, y in zip(weights, res_factors)])
        return r

    '''
    Save the weight vector and the base matrices. 
    '''
    def save_parametrization(self, base_matrices, weights, two_files, file_dir=None):
        rds = ResDataSaver(weights=weights, factor_params=base_matrices, filenames=two_files[1:], file_dir=file_dir)
        rds()

    def __call__(self, *args, **kwargs):
        res_grid = self.generate_resistance_grid(self.res_grid_shape)
        #print(res_grid)
        x_new, y_new = self.generate_elliptical_mapping(self.angle, self.sigma_x, self.sigma_y, self.mean_x, self.mean_y,
                                         seed=self.seed)
        #print('Starting Point')
        #print(self.starting_point)
        structured_res_grid = self.structured_resistance_grid(res_grid, x_new, y_new, self.starting_point,
                                                              self.ending_point)
        print(structured_res_grid)
        print(self.filenames)
        self.save_resistance_grid(structured_res_grid, self.filenames, file_dir=self.file_dir)
        res_matrix = self.resistance_grid_to_res_matrix(structured_res_grid, self.res_grid_shape)
        #print(res_matrix)
        cond_matrix = self.res_matrix_to_cond_matrix(res_matrix)
        #print(cond_matrix)
        b_m_list, w_vector = self.generate_cond_parametrization(cond_matrix, self.num_of_weights)
        #print('Weight vector:')
        #print(w_vector)
        #print('Base matrices:')
        #print(b_m_list)
        computed_cond_matrix = self.compute_res_matrix(w_vector, b_m_list)
        print('Test closeness:')
        print(np.allclose(cond_matrix, computed_cond_matrix))
        self.save_parametrization(b_m_list, w_vector, self.filenames, file_dir=self.file_dir)


if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Generate structured landscape resistance instances.")
    parser.add_argument("res_grid_shape", help="The shape of the ASCII resistance grid. Type the same value twice.",
                        nargs=2, type=int)
    parser.add_argument('starting_point', help="Type in coordinate pair of the start node for animal walks.", nargs=2,
                        type=int)
    parser.add_argument('ending_point', help="Type in coordinate pair of destination node for animal walks.", nargs=2,
                        type=int)
    parser.add_argument("num_of_weights", help="Size of weight vector in resistance factor parameter calculation.",
                        nargs='?', type=int)
    parser.add_argument("angle", help="The angle of the 2-D Gaussian function.", nargs='?', type=float)
    parser.add_argument("sigma_x", help="The standard deviation in the x-axis.", nargs='?', type=float)
    parser.add_argument("sigma_y", help="The standard deviation in the y-axis.", nargs='?', type=float)
    parser.add_argument("--mean_x", help="Specify the mean in the x-axis. Automatically set to the middle of the "
                                         "resistance grid.", dest="mean_x", nargs='?', type=float, default=None)
    parser.add_argument("--mean_y", help="Specify the mean in the y-axis. Automatically set to the middle of the "
                                         "resistance grid.", dest="mean_y", nargs='?', type=float, default=None)
    parser.add_argument('--filenames', help="List of filenames to store resistance matrix data. List one filename for "
                                            "storing resistance grid and two filenames for storing weights and factor "
                                            "parameters.", dest="filenames", default=None, const=None, nargs='*')
    parser.add_argument('--file_dir', help="File directory to store resistance matrix data. If not specified, "
                                           "--filenames argument must contain relative or absolute file paths.",
                        dest="dir", default=None, const=None, nargs='?')
    parser.add_argument("--seed", help="Seed of random number generator, automatically defaults to 42.", action='store',
                        default=42, dest='seed', nargs='?', const=42, type=int)
    args = parser.parse_args()
    srg = StructuredResGenerator(res_grid_shape=tuple(args.res_grid_shape), num_of_weights=args.num_of_weights,
                                 angle=args.angle, sigma_x=args.sigma_x, sigma_y=args.sigma_y, mean_x=args.mean_x,
                                 mean_y=args.mean_y, filenames=args.filenames, file_dir=args.dir, seed=args.seed,
                                 starting_point=args.starting_point, ending_point=args.ending_point)
    srg()
