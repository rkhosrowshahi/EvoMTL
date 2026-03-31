import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsArchitecture(nn.Module):
    r"""An abstract class for MTL architectures.

    Args:
        task_name (list): A list of strings for all tasks.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        device (torch.device): The device where model and data will be allocated. 
        kwargs (dict): A dictionary of hyperparameters of architectures.
     
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder_class = encoder_class
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs
        
        if self.rep_grad:
            self.rep_tasks = {}
            self.rep = {}
    
    def forward(self, inputs, task_name=None):
        r"""

        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        s_rep = self.encoder(inputs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out
    
    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()
    
    def get_all_params(self):
        r"""Return all parameters of the model.
        """
        all_params = None
    
        all_params = []
        for name, param in self.encoder.named_parameters():
            all_params.append((f'encoder.{name}', param))
        for task in self.task_name:
            for name, param in self.decoders[task].named_parameters():
                all_params.append((f'decoders.{task}.{name}', param))
        return all_params
    
    def set_all_params(self, params):
        r"""Set all parameters of the model.
        """
        # Set encoder parameters
        encoder_params = []
        decoder_params = {}
        
        # Split params into encoder and decoder based on param names
        for name, param in params:
            if name.startswith('encoder.'):
                encoder_params.append((name[8:], param))
            else:
                task = name.split('.')[1]
                if task not in decoder_params:
                    decoder_params[task] = []
                decoder_params[task].append((name.split('.', 2)[2], param))
                
        # Load parameters into models
        for name, param in encoder_params:
            state_dict = self.encoder.state_dict()
            state_dict[name].copy_(param)
            
        for task in self.task_name:
            state_dict = self.decoders[task].state_dict()
            for name, param in decoder_params[task]:
                state_dict[name].copy_(param)

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad(set_to_none=False)
        
    def _prepare_rep(self, rep, task, same_rep=None):
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep
