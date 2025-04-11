import random, torch, os
import numpy as np
import torch.nn as nn

def get_root_dir():
    r"""Return the root path of project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def set_device(gpu_id):
    r"""Set the device where model and data will be allocated. 

    Args:
        gpu_id (str, default='0'): The id of gpu.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def count_parameters(model):
    r'''Calculate the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('='*40)
    print('Total Params:', trainable_params + non_trainable_params)
    print('Trainable Params:', trainable_params)
    print('Non-trainable Params:', non_trainable_params)
        
def count_improvement(base_result, new_result, weight):
    r"""Calculate the improvement between two results as

    .. math::
        \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T 
        \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{B_{t,m}}.

    Args:
        base_result (dict): A dictionary of scores of all metrics of all tasks.
        new_result (dict): The same structure with ``base_result``.
        weight (dict): The same structure with ``base_result`` while each element is binary integer representing whether higher or lower score is better.

    Returns:
        float: The improvement between ``new_result`` and ``base_result``.

    Examples::

        base_result = {'A': [96, 98], 'B': [0.2]}
        new_result = {'A': [93, 99], 'B': [0.5]}
        weight = {'A': [1, 0], 'B': [1]}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        improvement += (((-1)**np.array(weight[task]))*\
                        (np.array(base_result[task])-np.array(new_result[task]))/\
                         np.array(base_result[task])).mean()
        count += 1
    return improvement/count

def set_param(curr_mod, name, param=None, mode='update'):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p
            

def build_model_from_blocks_using_gaussian(model, W, total_weights, solution, codebook, state, weight_offsets, device='cuda'):
        """
        Build model using the solution parameters from distribution based strategy
        Args:
            model: Base neural network model
            W: Number of components
            total_weights: Total number of parameters
            solution: Solution vector from distribution based strategy
            codebook: Dictionary mapping components to parameter indices
            state: Distribution based strategy state
            weight_offsets: Random offsets for parameters
            device: Device to place model on
        Returns:
            model: Updated model with new parameters
        """
        solution = np.array(solution)
        means = solution[:W]
        log_sigmas = solution[W:]
        sigmas = np.exp(log_sigmas)
        means = np.nan_to_num(means, nan=0)
        sigmas = np.nan_to_num(sigmas, nan=0)

        # Initialize parameter vector
        params = torch.zeros(total_weights, device=device)
        for k in range(W):
            indices = codebook[k]
            size = len(indices)
            if size > 0: 
                mean_tensor = torch.tensor(means[k], device=device)
                sigma_tensor = torch.tensor(sigmas[k], device=device)
                
                params[indices] = torch.normal(
                    mean=mean_tensor,
                    std=sigma_tensor,
                    size=(size,),
                    device=device
                ) # * weight_offsets[indices]
        # Assign weights to model
        torch.nn.utils.vector_to_parameters(params, model.parameters())

        return model

def build_model_from_blocks_using_centers(model, W, total_weights, solution, codebook, state, weight_offsets, device='cuda'):
        """
        Build model using the solution parameters from distribution based strategy
        Args:
            model: Base neural network model
            W: Number of components
            total_weights: Total number of parameters
            solution: Solution vector from distribution based strategy
            codebook: Dictionary mapping components to parameter indices
            state: Distribution based strategy state
            weight_offsets: Random offsets for parameters
            device: Device to place model on
        Returns:
            model: Updated model with new parameters
        """
        solution = np.array(solution)
        means = solution[:W]
        means = np.nan_to_num(means, nan=0)

        # Initialize parameter vector
        params = torch.zeros(total_weights, device=device)
        for k in range(W):
            indices = codebook[k]
            size = len(indices)
            if size > 0: 
                mean_tensor = torch.tensor(means[k], device=device)
                params[indices] = torch.full_like(params[indices], mean_tensor) # * weight_offsets[indices]
        # Assign weights to model
        torch.nn.utils.vector_to_parameters(params, model.parameters())

        return model


def ubp_cluster(W, params):
    """
    Uniform bin partitioning clustering
    Args:
        W: Number of bins/clusters
        params: Parameters to cluster
    Returns:
        codebook: Dictionary mapping cluster indices to parameter indices
        centers: Cluster centers
        bin_indices: Cluster assignments for each parameter
    """
    # Calculate bin edges
    min_val = params.min()
    max_val = params.max()
    bins = np.linspace(min_val, max_val, W)
    bin_indices = np.digitize(params, bins) - 1
    
    # Create codebook and compute centers
    centers = []
    log_sigmas = []
    counter = 0
    codebook = {}
    for i in range(W):
        mask = np.where(bin_indices == i)[0]
        if len(mask) == 0:
            continue
        centers.append(params[mask].mean())
        log_sigmas.append(np.log(params[mask].std() + 1e-8))
        bin_indices[mask] = counter
        codebook[counter] = mask

        counter+=1
    centers = np.array(centers)
    log_sigmas = np.array(log_sigmas)
    # Replace NaN values in log_sigmas with log(0.01) as default
    # log_sigmas = np.nan_to_num(log_sigmas, nan=0)
    return codebook, centers, log_sigmas, bin_indices


def plot_pareto_front_and_population(pf_F, pop_F, gd_point=None, iter=None, save_path=None, loss_names=None):
    import matplotlib.pyplot as plt
    if len(loss_names) == 2:
        plt.figure()
        
        if pop_F is not None:
            plt.scatter(pop_F[:, 0], pop_F[:, 1], facecolors='none', edgecolors='blue', label="Population")
        if pf_F is not None:
            plt.scatter(pf_F[:, 0], pf_F[:, 1], color="red", label="Pareto Front")
        
        if gd_point is not None:
            plt.scatter(gd_point[0], gd_point[1], color="green", label="GD")
        plt.xlabel(f'{loss_names[0]}')
        plt.ylabel(f'{loss_names[1]}')
        # plt.show()
        plt.legend()
        plt.savefig(save_path+f"/pareto_front_step{iter}.pdf")
        plt.close()
    elif len(loss_names) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if pop_F is not None:
            ax.scatter(pop_F[:, 0], pop_F[:, 1], pop_F[:, 2], facecolors='none', edgecolors='blue', label="Population", marker='o', s=50)
        if pf_F is not None:
            ax.scatter(pf_F[:, 0], pf_F[:, 1], pf_F[:, 2], color="red", label="Pareto Front", marker='o', s=10)
        
        if gd_point is not None:
            ax.scatter(gd_point[0], gd_point[1], gd_point[2], color="green", label="GD")
        # plt.show()
        ax.set_xlabel(f'{loss_names[0]}')
        ax.set_ylabel(f'{loss_names[1]}')
        ax.set_zlabel(f'{loss_names[2]}')
        ax.legend()
        fig.savefig(save_path+f"/pareto_front_step{iter}.pdf")
        plt.close()