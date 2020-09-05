import torch
from torch import nn
from InverseLearning.InvLearnCircuitscape import MatrixInverseGradients as mig


class InvLearnDirectLaplacian(nn.Module):

    def __init__(self, dimension, cond_matrix=None, weight_vector=None, base_matrices=None):
        super(InvLearnDirectLaplacian, self).__init__()
        self.dimension = dimension
        if base_matrices is not None:
            self.conductance_layer = ConductanceToLaplacian(self.dimension, cond_matrix=cond_matrix, weight_vector=weight_vector,
                                                        base_matrices=base_matrices)
        else:
            self.conductance_layer = ConductanceToLaplacian(self.dimension, cond_matrix=cond_matrix)
        self.ones_vector = OnesVector(self.dimension)
        self.current_vector = CurrentVector(self.dimension)
        self.inv = InvGraphLaplacian()

    def round_tensor(self, arr, n_digits):
        return torch.round(arr * 10 ** n_digits) / (10 ** n_digits)

    def contract_graph_lap(self, graph_lap, end_camera_trap):
        graph_lap[end_camera_trap] = float('Inf')
        graph_lap[:, end_camera_trap] = float('Inf')
        return graph_lap

    def print_weights_vector(self):
        return self.conductance_layer.print_weight_vector()

    def forward(self, first_cam_trap, last_cam_trap):
        #print('Laplacian forward starting')
        # Mini-batch under construction
        graph_lap, cond_mat = self.conductance_layer()
        ones_vec = self.ones_vector(first_cam_trap, last_cam_trap)
        #print('Conductance matrix gradient:')
        #print(cond_mat.requires_grad)
        #print('Ones Vector: ', ones_vec)
        #print('')
        current_vec = self.current_vector(first_cam_trap, last_cam_trap, cond_mat)
        #print('Current Vector gradient InvLearnLaplacian: ')
        #print('')
        #print(current_vec.requires_grad)
        #print('Graph Laplacian Full:')
        #print(graph_lap)
        #print('')
        inv_graph_lap = self.inv(self.contract_graph_lap(graph_lap, last_cam_trap))
        #print('Inverse Graph Laplacian')
        #print(inv_graph_lap.requires_grad)
        #print('')
        voltage_vector = torch.mm(inv_graph_lap, current_vec)
        #print('Voltage Vector')
        #print(voltage_vector)
        #print('')
        h_pred = torch.mm(ones_vec, voltage_vector)
        # Uncomment return once done with testing
        #return h_pred

        # Debugging purposes, delete once done
        #return h_pred, voltage_vector, ones_vec
        return h_pred


        '''
        # Old code

        graph_lap, cond_mat = self.conductance_layer()
        ones_vec = self.ones_vector(first_cam_trap, last_cam_trap)
        #print('Ones Vector')
        #print(ones_vec)
        current_vec = self.current_vector(first_cam_trap, last_cam_trap, cond_mat)
        #graph_lap_rounded = self.round_tensor(graph_lap, n)
        #print('Current Vector')
        #print(current_vec)
        #print(current_vec.requires_grad)
        print('Graph Laplacian')
        print(graph_lap)

        # Remove last_cam_trap argument for old code
        inv_graph_lap = self.inv(graph_lap)

        print('Inverse Graph Laplacian')
        print(inv_graph_lap)
        #print(inv_graph_lap.requires_grad)
        #print('Voltage vector')
        #print(torch.mm(inv_graph_lap, current_vec))
        h_pred = torch.mm(ones_vec, torch.mm(inv_graph_lap, current_vec))
        #print(h_pred.requires_grad)
        #return h_pred, inv_graph_lap, graph_lap, cond_mat, ones_vec, current_vec
        return h_pred
        '''


# Old Module
class OnesVector(nn.Module):
    def __init__(self, row_dim):
        super(OnesVector, self).__init__()
        self.row_dim = row_dim

    def forward(self, first_cam_trap, last_cam_trap):
        ones_vec = torch.zeros((1, self.row_dim), dtype=torch.float64)
        ones_vec[0, first_cam_trap] = 1.0
        ones_vec[0, last_cam_trap] = -1.0
        return ones_vec


'''
class OnesVector(nn.Module):
    def __init__(self, row_dim):
        super(OnesVector, self).__init__()
        self.row_dim = row_dim

    def forward(self, first_cam_trap):
        ones_vec = torch.zeros((1, self.row_dim-1), dtype=torch.float64)
        ones_vec[0, first_cam_trap] = 1.0
        return ones_vec
'''


# Old Module
class CurrentVector(nn.Module):
    def __init__(self, col_dim):
        super(CurrentVector, self).__init__()
        self.col_dim = col_dim

    def forward(self, first_cam_trap, last_cam_trap, cond_mat):
        row_sums = (cond_mat.sum(dim=1).reshape(-1, 1)).requires_grad_(requires_grad=True)
        row_sums[last_cam_trap] = 0.0
        row_sums[last_cam_trap] = -1.0*row_sums.sum()
        return row_sums


'''
class CurrentVector(nn.Module):
    def __init__(self, col_dim):
        super(CurrentVector, self).__init__()
        self.col_dim = col_dim

    def forward(self, first_cam_trap, last_cam_trap, cond_mat):
        row_sums = cond_mat.sum(dim=1).reshape(-1, 1)
        row_sums[last_cam_trap] = -1.0
        return torch.masked_select(row_sums, row_sums != -1.0).view(-1, 1)
'''


class ConductanceToLaplacian(nn.Module):
    def __init__(self, dimension, cond_matrix=None, weight_vector=None, base_matrices=None):
        super(ConductanceToLaplacian, self).__init__()
        self.dimension = dimension
        self.fixed_weights = self.generate_weight_matrix()
        torch.manual_seed(42)
        if base_matrices is not None:
            self.base_matrices = torch.from_numpy(base_matrices).type(torch.float64)
            #print(self.base_matrices)
        else:
            self.base_matrices = base_matrices
        # Parameter class called to add this tensor automatically to list of module parameters
        # When nn.Module.parameters() function is called, these set of parameters will be returned.
        if base_matrices is not None:
            if weight_vector is None:
                self.weights_vector = nn.Parameter(torch.rand(len(base_matrices), 1, dtype=torch.float64))
            else:
                self.weights_vector = nn.Parameter(torch.from_numpy(weight_vector).type(torch.float64))
            #self.conductance_matrix = torch.zeros(self.dimension, self.dimension, dtype=torch.float64)
            #for idx in range(len(self.weights_vector)):
                #self.conductance_matrix += self.weights_vector[idx]*base_matrices[idx]
            #print('Initial Conductance Matrix:')
            #print(self.conductance_matrix)
            print('Starting Weight Vector: ')
            print(self.weights_vector)
        else:
            if cond_matrix is not None:
                self.conductance_matrix = nn.Parameter(torch.from_numpy(cond_matrix).type(torch.float64))
                #print(self.conductance_matrix)
            else:
                self.conductance_matrix = nn.Parameter(self.initialize_distribution())
                #print(self.conductance_matrix)

    def initialize_distribution(self):
        a = torch.rand(self.dimension, self.dimension, dtype=torch.float64)
        a.retain_grad()
        return a

    def generate_weight_matrix(self):
        weight_matrix = torch.diag(-1.0*torch.ones(self.dimension*self.dimension, dtype=torch.float64))
        grid_range = torch.arange(start=0, end=self.dimension*self.dimension, step=self.dimension + 1)
        j = 0
        for i in grid_range:
            weight_matrix[i, j:(j+self.dimension)] = 1.0
            j = j + self.dimension
        return weight_matrix

    def print_weight_vector(self):
        return self.weights_vector

    def forward(self):
        if self.base_matrices is not None:
            self.conductance_matrix = torch.zeros(self.dimension, self.dimension, dtype=torch.float64)
            for idx in range(len(self.weights_vector)):
                #print('Weight vector:')
                #print(self.weights_vector[idx])
                #print('Base matrix:')
                #print(self.base_matrices[idx, :, :])
                self.conductance_matrix += self.weights_vector[idx]*self.base_matrices[idx, :, :]
            self.conductance_matrix = (self.conductance_matrix + self.conductance_matrix.t())/2
            #print('Conductance matrix:')
            #print(self.conductance_matrix)
        shape = self.conductance_matrix.shape
        cond_vector = torch.flatten(self.conductance_matrix).reshape(-1, 1)
        row_laplacian = torch.mm(self.fixed_weights, cond_vector)
        graph_laplacian = row_laplacian.reshape(shape)
        #print('Graph Laplacian')
        #print(graph_laplacian.requires_grad)
        #print(graph_laplacian)
        return graph_laplacian, self.conductance_matrix

'''
# Old module

class InvGraphLaplacian(nn.Module):
    def __init__(self):
        super(InvGraphLaplacian, self).__init__()
        self.laplacian_func = mig.KroneckerGradient.apply

    def forward(self, graph_lap):
        return self.laplacian_func(graph_lap)
'''


class InvGraphLaplacian(nn.Module):
    def __init__(self):
        super(InvGraphLaplacian, self).__init__()
        self.laplacian_func = mig.OptKroneckerGradient.apply

    def forward(self, graph_lap):
        return self.laplacian_func(graph_lap)








