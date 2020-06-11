import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def get_loss_function(loss_name):
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError


def perform_epoch(model, loader, optimizer=None, max_batch_count=None, device='cpu', loss_name='ce'):
    loss_function = get_loss_function(loss_name)
    
    cum_loss = 0
    cum_acc = 0
    cum_batch_size = 0
    batch_count = 0
    
    with torch.no_grad() if optimizer is None else torch.enable_grad():
        #for X, y in tqdm(loader):
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            batch_size = X.shape[0]

            logits = model(X)
            if loss_name == 'bce':
                logits = logits.view(-1)
                acc = torch.mean((logits * (y*2-1) > 0).float())
                loss = loss_function(logits, y.float())
            else:
                acc = torch.mean((torch.max(logits, dim=-1)[1] == y).float())
                loss = loss_function(logits, y)
                
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cum_loss += loss.item() * batch_size
            cum_acc += acc.item() * batch_size
            cum_batch_size += batch_size
            batch_count += 1
            
            if max_batch_count is not None and batch_count >= max_batch_count:
                break

    mean_loss = cum_loss / cum_batch_size
    mean_acc = cum_acc / cum_batch_size

    return mean_loss, mean_acc


def get_tangent_kernels(model, loader, optimizer, batch_size=None, device='cpu'):
    for hid_layer in model.hidden_layers:
        if hasattr(hid_layer, 'weight'):
            raise NotImpementedError
    
    sample_count = 0
    stop_flag = False
    
    input_ntk_diag = []
    output_ntk_diag = []

    for X, _ in loader:
        X = X.to(device)

        logits = model(X)
        if logits.shape[-1] > 1:
            raise NotImpementedError
        else:
            logits = logits.view(-1)

        for logit in logits:
            grads_at_x = torch.autograd.grad([logit], [model.input_layer.weight, model.output_layer.weight], retain_graph=True)
            input_ntk_diag.append(torch.sum(grads_at_x[0] ** 2)  * optimizer.param_groups[0]['lr'])
            output_ntk_diag.append(torch.sum(grads_at_x[1] ** 2) * optimizer.param_groups[2]['lr'])
            
            sample_count += 1
            if batch_size is not None and sample_count >= batch_size:
                stop_flag = True
                break
                
        if stop_flag:
            break

    input_ntk_diag = torch.stack(input_ntk_diag).cpu().numpy()
    output_ntk_diag = torch.stack(output_ntk_diag).cpu().numpy()

    return input_ntk_diag, None, output_ntk_diag


def get_logits(model, loader, max_batch_count=None, device='cpu'):
    
    logits = []
    batch_count = 0
    
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            logits.append(model(X).cpu())
            
            batch_count += 1
            if max_batch_count is not None and batch_count >= max_batch_count:
                break
            
        logits = torch.cat(logits, dim=0)
        logits = logits.numpy()

    return logits