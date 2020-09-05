import torch
from torch import nn
import math


class InvLearnConjugateGradient(nn.Module):
    def __init__(self, dimension, cond_matrix=None, weight_vector=None, base_matrices=None, tol=0.001):
        super(InvLearnConjugateGradient, self).__init__()
        self.n = dimension
        if base_matrices is not None:
            self.conductance_layer = ConductanceToLaplacianCG(self.n, cond_matrix=cond_matrix, weight_vector=weight_vector,
                                                          base_matrices=base_matrices)
        else:
            self.conductance_layer = ConductanceToLaplacianCG(self.n, cond_matrix=cond_matrix)
        self.ones_vector = OnesVectorCG(self.n)
        self.current_vector = CurrentVectorCG(self.n)
        self.cg = CGOneStep()
        self.tol = tol
        self.starting_vector = None

    def print_weights_vector(self):
        return self.conductance_layer.print_weight_vector()

    def return_dimension(self):
        return self.n

    def contract_graph_lap(self, graph_lap, end_camera_trap):
        graph_lap[end_camera_trap] = float('Inf')
        graph_lap[:, end_camera_trap] = float('Inf')
        indices = (graph_lap != float('Inf'))
        s = graph_lap.shape
        return torch.masked_select(graph_lap, indices).view(s[0] - 1, s[1] - 1)

    def expand_voltage_vector(self, volt_vector, last_cam_trap):
        v_vec = torch.zeros(len(volt_vector)+1, 1, dtype=torch.float64)
        v_vec[last_cam_trap] = float('Inf')
        indices = (v_vec != float('Inf'))
        v_vec[indices] = volt_vector.view(-1)
        v_vec[last_cam_trap] = 0.0
        return v_vec

    def forward(self, first_cam_trap, last_cam_trap, initial_guess=None, full=64):
        #print('Conjugate gradient forward starting')
        # Old conjugate gradient code
        '''
        graph_lap, cond_mat = self.conductance_layer()
        ones_vec = self.ones_vector(first_cam_trap, last_cam_trap)
        current_vec = self.current_vector(first_cam_trap, last_cam_trap, cond_mat)
        if initial_guess is None:
            x_i = torch.zeros((graph_lap.shape[0], 1), dtype=torch.float64)
        else:
            x_i = initial_guess
        r_i = current_vec - torch.mm(graph_lap, x_i)
        if r_i.norm() < self.tol:
            return torch.mm(ones_vec, x_i)
        else:
            d_i = current_vec - torch.mm(graph_lap, x_i)
            i = 0
            while i < self.n:
                if r_i.norm() > self.tol:
                    i = i + 1
                    x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
                else:
                    i = self.n + 1
        return torch.mm(ones_vec, x_i), x_i, ones_vec
        '''
        #start_cg = time.perf_counter()
        # Optimized conjugate gradient
        graph_lap_full, cond_mat = self.conductance_layer()
        #print('Conductance matrix:')
        #print(cond_mat)
        graph_lap = self.contract_graph_lap(graph_lap_full, last_cam_trap)
        #print(graph_lap)
        ones_vec = self.ones_vector(first_cam_trap, last_cam_trap)
        current_vec = self.current_vector(first_cam_trap, last_cam_trap, cond_mat)

        #end_cg_1 = time.perf_counter()
        #print('Non CG Time: ',  end_cg_1 - start_cg)

        #print('CG Forward starting')
        #print('Check gradient on current vector:')
        #print(current_vec)
        if initial_guess is None and self.starting_vector is None:
            x_i = torch.zeros((graph_lap.shape[0], 1), dtype=torch.float64)
        elif initial_guess is not None:
            x_i = initial_guess
        elif self.starting_vector is not None:
            x_i = self.starting_vector

        #print('Starting Vector:')
        #print(x_i.t())
        r_i = current_vec - torch.mm(graph_lap, x_i)
        #print('Check gradient on r_i')
        #print(r_i)
        #print('r_i norm: ', r_i.norm())
        if r_i.norm() < self.tol:
            # When the r_i.norm() is 0, it will cause zero by division error
            # Doing another set of CG will yield NAN values.
            # Experimental
            epsilon = 0.000000001
            r_i = r_i + epsilon
            d_i = current_vec - torch.mm(graph_lap, x_i)
            d_i = d_i + epsilon
            x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
            return torch.mm(ones_vec, self.expand_voltage_vector(x_i, last_cam_trap))
        else:
            d_i = current_vec - torch.mm(graph_lap, x_i)
            #print('Check gradient on d_i:')
            #print(d_i)
            i = 0
            '''
            if full is False:
                end = int(math.sqrt(self.n)) + 1
            else:
                end = self.n + 1
                #print('Starting x_i: ', x_i)
            '''
            end = full
            #print('CG Iterations: ', end)

            # New Module to force full iterations
            '''
            while i < end:
                #print('Iteration: ', i)
                i = i + 1
                x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
                #print('Norm: ', r_i)
                if i == end:
                    #print('Iterations: ', i)
                    self.starting_vector = x_i.clone().detach()
                    #i = i + 1
                    #print('Iterations: ', i)
            '''

            '''
            # Testing the New Module to force full iterations
            first = False
            while i < end:
                if r_i.norm() > self.tol:
                    i = i + 1
                    x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
                elif r_i.norm() <= self.tol and first is False:
                    first = True
                    print('Original Stopping Point: ', x_i.norm())
                    i = i + 1
                    x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
                else:
                    i = i + 1
                    x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
            self.starting_vector = x_i.clone().detach()
            print('Final r_i norm: ', x_i.norm())
            '''


            # Old Module basing if r_i.norm() < tol
            while i < end:
                if r_i.norm() > self.tol:
                    #print('Iteration: ', i)
                    i = i + 1
                    x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
                    #print('Norm: ', r_i)
                else:
                    #print('Iterations: ', i)
                    self.starting_vector = x_i.clone().detach()
                    i = self.n + 1

            #print('Iterations: ', i)
            #print('r_i norm: ', r_i.norm().item())
            #print('Final x: ', x_i.t())
            #return torch.mm(ones_vec, self.expand_voltage_vector(x_i, last_cam_trap)), \
                   #self.expand_voltage_vector(x_i, last_cam_trap), ones_vec, x_i
            #end_cg_2 = time.perf_counter()
            #print('CG Time: ', end_cg_2 - end_cg_1)
            #print('')
            return torch.mm(ones_vec, self.expand_voltage_vector(x_i, last_cam_trap))


class OnesVectorCG(nn.Module):
    def __init__(self, row_dim):
        super(OnesVectorCG, self).__init__()
        self.row_dim = row_dim

    def forward(self, first_cam_trap, last_cam_trap):
        ones_vec = torch.zeros((1, self.row_dim), dtype=torch.float64)
        ones_vec[0, first_cam_trap] = 1.0
        ones_vec[0, last_cam_trap] = -1.0
        return ones_vec


# Old Module
'''
class CurrentVectorCG(nn.Module):
    def __init__(self, col_dim):
        super(CurrentVectorCG, self).__init__()
        self.col_dim = col_dim

    def forward(self, first_cam_trap, last_cam_trap, cond_mat):
        row_sums = cond_mat.sum(dim=1).reshape(-1, 1)
        row_sums[last_cam_trap] = 0.0
        row_sums[last_cam_trap] = -1.0*row_sums.sum()
        return row_sums
'''


# Optimized CG Module
class CurrentVectorCG(nn.Module):
    def __init__(self, col_dim):
        super(CurrentVectorCG, self).__init__()
        self.col_dim = col_dim

    def forward(self, first_cam_trap, last_cam_trap, cond_mat):
        row_sums = cond_mat.sum(dim=1).reshape(-1, 1)
        #print('Row sums')
        #print(row_sums.requires_grad)
        row_sums[last_cam_trap] = -1.0
        return torch.masked_select(row_sums, row_sums != -1.0).view(-1, 1)


class ConductanceToLaplacianCG(nn.Module):
    def __init__(self, dimension, cond_matrix=None, weight_vector=None, base_matrices=None):
        super(ConductanceToLaplacianCG, self).__init__()
        self.dimension = dimension
        self.fixed_weights = self.generate_weight_matrix()
        torch.manual_seed(42)
        # Parameter class called to add this tensor automatically to list of module parameters
        # When nn.Module.parameters() function is called, these set of parameters will be returned.
        if base_matrices is not None:
            self.base_matrices = torch.from_numpy(base_matrices).type(torch.float64)
        else:
            self.base_matrices = base_matrices

        if base_matrices is not None:
            if weight_vector is None:
                self.weights_vector = nn.Parameter(torch.rand(len(base_matrices), 1, dtype=torch.float64))
                #self.conductance_matrix = torch.zeros(self.dimension, self.dimension, dtype=torch.float64)
                #for idx in range(len(self.weights_vector)):
                    #self.conductance_matrix += self.weights_vector[idx]*base_matrices[idx]
                #print('Initial Conductance Matrix:')
                #print(self.conductance_matrix)
            else:
                self.weights_vector = nn.Parameter(torch.from_numpy(weight_vector).type(torch.float64))
            print('Starting Weight Vector: ')
            print(self.weights_vector)
        else:
            if cond_matrix is not None:
                self.conductance_matrix = nn.Parameter(torch.from_numpy(cond_matrix))
                #print(self.conductance_matrix)
            else:
                self.conductance_matrix = nn.Parameter(self.initialize_distribution())
                #print(self.conductance_matrix)

    def initialize_distribution(self):
        a = torch.rand(self.dimension, self.dimension, dtype=torch.float64)
        a.retain_grad()
        return a

    def print_weight_vector(self):
        return self.weights_vector

    def generate_weight_matrix(self):
        weight_matrix = torch.diag(-1.0*torch.ones(self.dimension*self.dimension, dtype=torch.float64))
        grid_range = torch.arange(start=0, end=self.dimension*self.dimension, step=self.dimension + 1)
        j = 0
        for i in grid_range:
            weight_matrix[i, j:(j+self.dimension)] = 1.0
            j = j + self.dimension
        return weight_matrix

    # This is a new function
    def create_graph_laplacian(self, cond_matrix):
        diag = torch.diag(torch.sum(cond_matrix, dim=1))
        extra = torch.diag(torch.diag(cond_matrix))
        g_lap = diag - cond_matrix + extra
        return g_lap

    def forward(self):
        if self.base_matrices is not None:
            self.conductance_matrix = torch.zeros(self.dimension, self.dimension, dtype=torch.float64)
            for idx in range(len(self.weights_vector)):
                self.conductance_matrix += self.weights_vector[idx]*self.base_matrices[idx, :, :]
            self.conductance_matrix = (self.conductance_matrix + self.conductance_matrix.t())/2
        # Old Module

        #shape = self.conductance_matrix.shape
        #cond_vector = torch.flatten(self.conductance_matrix).reshape(-1, 1)
        #row_laplacian = torch.mm(self.fixed_weights, cond_vector)
        #graph_laplacian = row_laplacian.reshape(shape)


        # New Module
        graph_laplacian = self.create_graph_laplacian(self.conductance_matrix)
        #diff = torch.sum(torch.abs(graph_laplacian_1 - graph_laplacian))
        #print('Difference: ', diff)

        #print('Graph Laplacian')
        #print(graph_laplacian.requires_grad)
        #print(graph_laplacian)
        return graph_laplacian, self.conductance_matrix


class CGOneStep(nn.Module):
    def __init__(self):
        super(CGOneStep, self).__init__()

    def forward(self, L_g, x_i, r_i, d_i):
        #print('x_i')
        #print(x_i)
        #print('r_i')
        #print(r_i)
        #print('d_i')
        #print(d_i)
        alpha = torch.mm(r_i.t(), r_i)/torch.mm(d_i.t(), torch.mm(L_g, d_i))
        #print('alpha')
        #print(alpha)
        x_i2 = x_i + alpha*d_i
        r_i2 = r_i - alpha*torch.mm(L_g, d_i)
        beta = torch.mm(r_i2.t(), r_i2)/torch.mm(r_i.t(), r_i)
        #print('beta')
        #print(beta)
        d_i2 = r_i2 + beta*d_i
        return x_i2, r_i2, d_i2
