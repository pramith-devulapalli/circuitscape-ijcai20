import torch
from torch import nn


class MantelOptConjugateGradient(nn.Module):
    def __init__(self, dimension, weight_vector, base_matrices, tol=0.001):
        super(MantelOptConjugateGradient, self).__init__()
        torch.manual_seed(42)
        # N is the size of the graph Laplacian
        self.n = dimension
        self.weights_to_res = WeightsToResistanceCG(weight_vector, base_matrices, dimension)
        self.res_to_cond = ResistanceToConductanceCG()

        self.cg_full = CGFullIteration(self.n, tol=tol)

    def print_weights_vector(self):
        return self.weights_to_res.print_weight_vector()

    def forward(self, start_point, end_point, full_iterations=1):
        res_matrix = self.weights_to_res(self.n)
        cond_matrix = self.res_to_cond(res_matrix)
        J = torch.zeros(1, 1, dtype=torch.float64)

        J = self.cg_full(cond_matrix, start_point, end_point, full=full_iterations)
        return J


class WeightsToResistanceCG(nn.Module):
    def __init__(self, weight_vector, base_matrices, dimension):
        super(WeightsToResistanceCG, self).__init__()
        self.base_matrices = torch.from_numpy(base_matrices).type(torch.float64)
        self.weights_vector = nn.Parameter(torch.from_numpy(weight_vector).type(torch.float64))
        #self.resistance_matrix = torch.zeros(dimension, dimension, dtype=torch.float64)

    def print_weight_vector(self):
        return self.weights_vector

    def forward(self, dimension):
        #print(self.weights_vector[0].shape)
        #print('Base Matrices: ')
        #print(self.weights_vector*self.base_matrices[0, :, :])
        resistance_matrix = torch.zeros(dimension, dimension, dtype=torch.float64)

        for idx in range(len(self.weights_vector)):
            resistance_matrix += self.weights_vector[idx]*self.base_matrices[idx, :, :]

        return resistance_matrix


class ResistanceToConductanceCG(nn.Module):
    def __init__(self):
        super(ResistanceToConductanceCG, self).__init__()

    def forward(self, res_mat):
        res_mat[res_mat == 0] = float('Inf')
        return torch.reciprocal(res_mat)


class ConductanceToLaplacianCG(nn.Module):
    def __init__(self, dimension, seed=42):
        super(ConductanceToLaplacianCG, self).__init__()
        self.dimension = dimension
        torch.manual_seed(seed)

    def create_graph_laplacian(self, cond_matrix):
        diag = torch.diag(torch.sum(cond_matrix, dim=1))
        extra = torch.diag(torch.diag(cond_matrix))
        g_lap = diag - cond_matrix + extra
        return g_lap

    def forward(self, cond_matrix):
        graph_laplacian = self.create_graph_laplacian(cond_matrix)
        return graph_laplacian


class OnesVectorCG(nn.Module):
    def __init__(self, row_dim):
        super(OnesVectorCG, self).__init__()
        self.r = row_dim

    def forward(self, first_cam_trap, last_cam_trap):
        ones_vec = torch.zeros((1, self.r), dtype=torch.float64)
        ones_vec[0, first_cam_trap] = 1.0
        ones_vec[0, last_cam_trap] = -1.0
        return ones_vec


class CurrentVectorCG(nn.Module):
    def __init__(self, col_dim):
        super(CurrentVectorCG, self).__init__()
        self.c = col_dim

    def forward(self, first_cam_trap, last_cam_trap):
        current_vec = torch.zeros((self.c, 1), dtype=torch.float64)
        current_vec[first_cam_trap, 0] = 1.0
        current_vec[last_cam_trap, 0] = -1.0
        return torch.masked_select(current_vec, current_vec != -1.0).view(-1, 1)


class CGOneStep(nn.Module):
    def __init__(self):
        super(CGOneStep, self).__init__()

    def forward(self, L_g, x_i, r_i, d_i):
        alpha = torch.mm(r_i.t(), r_i)/torch.mm(d_i.t(), torch.mm(L_g, d_i))
        x_i2 = x_i + alpha*d_i
        r_i2 = r_i - alpha*torch.mm(L_g, d_i)
        beta = torch.mm(r_i2.t(), r_i2)/torch.mm(r_i.t(), r_i)
        d_i2 = r_i2 + beta*d_i
        return x_i2, r_i2, d_i2


class CGFullIteration(nn.Module):
    def __init__(self, n, tol=0.001):
        super(CGFullIteration, self).__init__()
        self.n = n
        self.conductance_layer = ConductanceToLaplacianCG(self.n)
        self.ones_vector = OnesVectorCG(self.n)
        self.current_vector = CurrentVectorCG(self.n)
        self.cg = CGOneStep()
        self.tol = tol
        self.starting_vector = None

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
        v_vec[last_cam_trap] = 0
        return v_vec

    def forward(self, cond_matrix, first_cam_trap, last_cam_trap, initial_guess=None, full=1):
        graph_lap_full = self.conductance_layer(cond_matrix)
        graph_lap = self.contract_graph_lap(graph_lap_full, last_cam_trap)
        ones_vec = self.ones_vector(first_cam_trap, last_cam_trap)
        current_vec = self.current_vector(first_cam_trap, last_cam_trap)

        if initial_guess is None and self.starting_vector is None:
            x_i = torch.zeros((graph_lap.shape[0], 1), dtype=torch.float64)
        elif initial_guess is not None:
            x_i = initial_guess
        elif self.starting_vector is not None:
            x_i = self.starting_vector

        r_i = current_vec - torch.mm(graph_lap, x_i)
        #print('Initial r_i norm: ', r_i.norm())
        #cg_ave_iterations = 0

        if r_i.norm() < self.tol:
            d_i = current_vec - torch.mm(graph_lap, x_i)
            x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
            # Returning the commuting time, not hitting time
            return torch.sum(cond_matrix)*torch.mm(ones_vec, self.expand_voltage_vector(x_i, last_cam_trap))
            #return torch.sum(cond_matrix)*torch.mm(ones_vec, x_i)
        else:
            d_i = current_vec - torch.mm(graph_lap, x_i)
            i = 0
            #cg_ave_iterations = full
            '''
            if full is False:
                end = int(math.sqrt(self.n)) + 1
            else:
                end = self.n + 1
            '''
            end = full
            #print('CG Iterations: ', end)
            while i < end:
                if r_i.norm() > self.tol:
                    i = i + 1
                    #print('i: ', i+1)
                    x_i, r_i, d_i = self.cg(graph_lap, x_i, r_i, d_i)
                else:
                    #print('i: ', i+1)
                    #print(r_i.norm())
                    self.starting_vector = x_i.clone().detach()
                    cg_ave_iterations = i
                    i = self.n + 1
        # Returning the commuting time, not hitting time
        return torch.sum(cond_matrix)*torch.mm(ones_vec, self.expand_voltage_vector(x_i, last_cam_trap))
        #return torch.sum(cond_matrix)*torch.mm(ones_vec, x_i)