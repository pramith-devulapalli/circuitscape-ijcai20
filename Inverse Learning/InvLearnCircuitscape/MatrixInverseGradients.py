import torch
from torch.autograd import Function


class KroneckerGradient(Function):

    '''
    Input Parameters:
    ctx - torch.autograd context object
    *args - Positional arguments in the following order:
          - args[0] - Graph laplacian
    **kwargs - N/A
    '''
    @staticmethod
    def forward(ctx, *args, **kwargs):
        graph_lap_inv = torch.pinverse(args[0])
        ctx.save_for_backward(graph_lap_inv)
        return graph_lap_inv

    '''
    Input Parameters:
    ctx - torch.autograd context object
    **kwargs - N/A
    '''
    @staticmethod
    def backward(ctx, **kwargs):
        graph_lap_inv, = ctx.saved_tensors
        n = graph_lap_inv.shape[0]
        graph_lap_inv_tiled = torch.repeat_interleave(torch.repeat_interleave(graph_lap_inv, n, dim=0), n, dim=1)
        kron_prod = -1.0 * torch.mul(graph_lap_inv_tiled, graph_lap_inv_tiled.t())
        return kron_prod


class OptKroneckerGradient(Function):

    '''
    Input Parameters:
    ctx - torch.autograd context object
    *args - Positional arguments in the following order:
          - args[0] - Graph laplacian
    **kwargs - N/A
    '''
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # Experimental Code
        # copy from here
        graph_lap_full = args[0]
        #print('Graph Lap Full: ')
        #print(graph_lap_full)
        s = graph_lap_full.shape
        indices = (graph_lap_full != float('Inf'))
        graph_lap = torch.masked_select(graph_lap_full, indices).view(s[0] - 1, s[1] - 1)

        # copy till here

        #print('Graph Lap Cropped:')
        #print(graph_lap)
        graph_lap_inv = torch.pinverse(graph_lap)
        #print('Inv Graph Lap Cropped')
        #print(graph_lap_inv)
        
        d = graph_lap_full[0]
        g = torch.arange(len(d))
        end_cam_trap = 0
        for i in g:
            if d[i] == float('Inf'):
                end_cam_trap = i
                break
        #print('End cam trap')
        #print(end_cam_trap)
        
        graph_lap_inv_full = torch.ones(s[0], s[1], dtype=torch.float64)
        graph_lap_inv_full[end_cam_trap] = float('Inf')
        graph_lap_inv_full[:, end_cam_trap] = float('Inf')
        ind = graph_lap_inv_full != float('Inf')
        ind_inf = graph_lap_inv_full == float('Inf')
        graph_lap_inv_full[ind] = graph_lap_inv.view(-1)
        graph_lap_inv_full[ind_inf] = 0.0
        #print('Graph lap inv full shape')
        #print(graph_lap_inv_full)

        ctx.save_for_backward(graph_lap_inv_full)
        return graph_lap_inv_full


    '''
    Input Parameters:
    ctx - torch.autograd context object
    '''
    @staticmethod
    def backward(ctx, grad_output):
        graph_lap_inv, = ctx.saved_tensors
        s = graph_lap_inv.shape
        dL_dLg = torch.zeros_like(graph_lap_inv)
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[0]):
                    for l in range(s[1]):
                        dL_dLg[i, j] = dL_dLg[i, j] + grad_output[k, l]*-1.0*graph_lap_inv[k, i]*graph_lap_inv[j, l]
                        #dL_dLg[k, l] = dL_dLg[k, l] + grad_output[i, j]*-1.0*graph_lap_inv[i, k]*graph_lap_inv[l, j]

        return dL_dLg

        # Optimized kronecker gradient code
        '''
        #loss_grad = grad_output.clone()
        stored_multiply = -1.0*torch.mul(grad_output, graph_lap_inv)
        sum_val = torch.sum(stored_multiply)
        opt_kron_grad = torch.mul(sum_val.repeat(stored_multiply.shape), graph_lap_inv.t())

        return opt_kron_grad
        '''


