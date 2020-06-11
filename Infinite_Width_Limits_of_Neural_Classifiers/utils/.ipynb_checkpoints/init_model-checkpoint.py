import numpy as np

import torch
import torch.optim as optim

from itertools import chain


def _test_gradient_normalization(optimizer):
    if isinstance(optimizer, (optim.SGD,)):
        return False
    elif isinstance(optimizer, (optim.Adadelta, optim.Adagrad, optim.Adam, optim.AdamW, optim.Adamax, optim.RMSprop)):
        return True
    else:
        raise ValueError("Unsupported optimizer: {}".format(type(optimizer)))


def scale_hyperparams(input_layer, hidden_layers, output_layer, 
                      optimizer, width_factor, scaling_mode):
    
    #assert len(optimizer.param_groups) > 1, "there should be at least two param groups in optimizer; the first one should correspond to input layer"
    #assert optimizer.param_groups[0]['params'] == input_layer.parameters(), "the first param group of optimizer should correspond to input layer parameters"
    
    all_except_input_layers = chain(hidden_layers, [output_layer])
    
    #is_gradient_normalized = _test_gradient_normalization(optimizer)
    
    if scaling_mode == 'default':
        weight_factor = 1
        lr_factor = 1
    
    elif scaling_mode == 'preserve_initial_logit_std':
        weight_factor = 1
        lr_factor = width_factor ** (-0.5)
        
    elif scaling_mode == 'preserve_logit_mean':
        weight_factor = width_factor ** (-0.5)
        lr_factor = width_factor ** (-1.)
        
    elif scaling_mode == 'preserve_dynamics':
        # likely, incorrect; don't use it
        weight_factor = 1
        lr_factor = width_factor ** (-1.)
        
    else:
        raise ValueError("Unknown scaling mode: {}".format(scaling_mode))
        
    for layer in all_except_input_layers:
        if hasattr(layer, 'weight'):
            layer.weight.data = layer.weight.data * weight_factor
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                raise NotImplementedError
                
    for param_group in optimizer.param_groups[1:]:
        param_group['lr'] *= lr_factor
    