from copy import copy, deepcopy
from time import time
from tqdm import tqdm, trange

import numpy as np
import torch

from utils.scale_hyperparams import scale_hyperparams
from utils.perform_epoch import perform_epoch, get_logits, get_tangent_kernels
from utils.models import LinearizedModel, OptimizerForLinearizedModel, CustomBatchNorm1d


class EarlyStopper():
    def __init__(self, stop_by_validation: bool, patience=10, max_training_time=None):
        self.stop_by_validation = stop_by_validation
        self.patience = patience
        self.steps_made = 0
        self.best_val_perf = None
        self.best_val_perf_step = -1
        
        self.max_training_time = max_training_time
        self.start_time = time()
        
        self.stop_reason = None
        
    def stop_condition(self, val_perf=None):
        current_time = time()
        if self.max_training_time is not None:
            if current_time - self.start_time > self.max_training_time:
                self.stop_reason = 'time'
                return True
            
        if self.stop_by_validation:
            if self.best_val_perf is None or val_perf > self.best_val_perf:
                self.best_val_perf = val_perf
                self.best_val_perf_step = self.steps_made
            if self.steps_made - self.best_val_perf_step > self.patience:
                self.stop_reason = 'val_perf'
                return True
        
        self.steps_made += 1
        return False


def get_model_with_modified_width(model_class, reference_model_kwargs, 
                                  width_arg_name='width', width_factor=1,
                                  device='cpu'):
    assert width_arg_name in reference_model_kwargs
    model_kwargs = deepcopy(reference_model_kwargs)
    model_kwargs[width_arg_name] = int(model_kwargs[width_arg_name] * width_factor)
    
    model = model_class(**model_kwargs).to(device)
    
    return model


def get_optimizer(optimizer_class, optimizer_kwargs, model):
    assert hasattr(model, 'input_layer')
    assert hasattr(model, 'hidden_layers')
    assert hasattr(model, 'output_layer')
    optimizer = optimizer_class(
        [
            {'params': model.input_layer.parameters()},
            {'params': model.hidden_layers.parameters()},
            {'params': model.output_layer.parameters()}
        ], **optimizer_kwargs
    )
    return optimizer


def get_model_decomposition_term(model, model_init, output_inc, hidden_incs, input_inc):
    with torch.no_grad():
        model_decomposition_term = deepcopy(model_init)
        
        if input_inc:
            model_decomposition_term.input_layer.weight.data = model.input_layer.weight.data - model_init.input_layer.weight.data
            
        if output_inc:
            model_decomposition_term.output_layer.weight.data = model.output_layer.weight.data - model_init.output_layer.weight.data
        
        for layer_decomp, layer, layer_init in zip(model_decomposition_term.hidden_layers, model.hidden_layers, model_init.hidden_layers):
            if isinstance(layer, CustomBatchNorm1d):
                layer_decomp = deepcopy(layer)
#                 assert layer.num_batches_tracked == layer_decomp.num_batches_tracked
#                 assert torch.all(layer.running_mean == layer_decomp.running_mean)
#                 assert torch.all(layer.running_var == layer_decomp.running_var)
            if hidden_incs:
                if hasattr(layer, 'weight'):
                    if layer.weight is not None:
                        layer_decomp.weight.data = layer.weight.data - layer_init.weight.data
                    
        return model_decomposition_term
    
    
def _get_work_model(model, model_init, model_init_before_scaling, scaling_mode):
    if not isinstance(scaling_mode, str):
        return model
    if scaling_mode.endswith('_simple_init_corrected'):
#         print('_get_work_model is correct')
        return lambda x: model(x) - model_init(x) + model_init_before_scaling(x)
    elif scaling_mode.endswith('_init_corrected'):
        return lambda x: model(x) - model_init(x, model_to_get_masks_from=model) + model_init_before_scaling(x)
    else:
        return model


def train_and_eval(model, model_init, optimizer, scaling_mode, train_loader, test_loader, test_loader_det, 
                   num_epochs, correction_epoch,
                   eval_every=None, test_batch_count=None, 
                   logits_batch_count=None, tangent_kernels_batch_size=None, track_logits=False, track_tangent_kernels=False,
                   width_factor=1, device='cpu', print_progress=False, binary=False, eval_model_only=False,
                   stop_by_validation=False, patience=10, max_training_time=None):
    train_losses = {}
    test_losses = {}
    
    train_accs = {}
    test_accs = {}
    
    test_logits = {}
    test_tangent_kernels = {}
    
    init_corrected = isinstance(scaling_mode, str) and scaling_mode.endswith('init_corrected')
#     print('init_corrected:', init_corrected)
    
    loss_name = 'bce' if binary else 'ce'
    
    if model is None:
        model = model_init
        
        if init_corrected:
            model_init_before_scaling = deepcopy(model_init)
        else:
            model_init_before_scaling = None
            
        stopper = EarlyStopper(stop_by_validation=stop_by_validation, patience=patience, max_training_time=max_training_time)

        epoch_iterator = trange(num_epochs)
        for epoch in epoch_iterator:
            if scaling_mode == 'linearized':
                if epoch == 0:
                    scale_hyperparams(
                        model.input_layer, model.hidden_layers, model.output_layer, 
                        optimizer=optimizer, width_factor=width_factor, scaling_mode=scaling_mode,
                        epoch=epoch, correction_epoch=correction_epoch
                    )
                    model = LinearizedModel(model)
                    optimizer = OptimizerForLinearizedModel(optimizer, model)
            else:
                scale_hyperparams(
                    model.input_layer, model.hidden_layers, model.output_layer, 
                    optimizer=optimizer, width_factor=width_factor, scaling_mode=scaling_mode,
                    epoch=epoch, correction_epoch=correction_epoch
                )

            if epoch == 0:
                model_init = deepcopy(model)
                work_model = _get_work_model(model, model_init, model_init_before_scaling, scaling_mode)

            if eval_every is not None and (epoch+1) % eval_every == 0:
                model.eval()
                test_loss, test_acc = perform_epoch(work_model, test_loader, device=device, max_batch_count=test_batch_count, loss_name=loss_name)
                test_losses[epoch] = test_loss
                test_accs[epoch] = test_acc
                if track_logits:
                    test_logits[epoch] = get_logits(work_model, test_loader_det, max_batch_count=logits_batch_count, device=device)
                if track_tangent_kernels:
                    test_tangent_kernels[epoch] = get_tangent_kernels(
                        model, test_loader_det, optimizer=optimizer, 
                        batch_size=tangent_kernels_batch_size, device=device
                    )
                if print_progress:
                    print('test_loss = {:.4f}; test_acc = {:.2f}'.format(test_loss, test_acc*100))
            else:
                test_loss, test_acc = None, None
    
            model.train()
            train_loss, train_acc = perform_epoch(work_model, train_loader, optimizer=optimizer, device=device, loss_name=loss_name)
            train_losses[epoch] = train_loss
            train_accs[epoch] = train_acc

            if print_progress:
                print('Epoch {};'.format(epoch+1))
                print('train_loss = {:.4f}; train_acc = {:.2f}'.format(train_loss, train_acc*100))
                
            if stopper.stop_condition(test_acc):
                print('Stopped by early stopper on epoch {}; reason: {}'.format(epoch, stopper.stop_reason))
                epoch_iterator.close()
                break

    elif model_init is None:
        raise ValueError("if model is not None, model_init should be not None too")
        
    training_time = time() - stopper.start_time

    model.eval()
    model_init.eval()
    start_inference_time = time()
    final_train_loss, final_train_acc = perform_epoch(work_model, train_loader, device=device, loss_name=loss_name)
    final_test_loss, final_test_acc = perform_epoch(work_model, test_loader, device=device, loss_name=loss_name)
    inference_time = time() - start_inference_time
    
    if track_logits:
        final_test_logits = get_logits(work_model, test_loader_det, device=device)
    else:
        final_test_logits = None
        
    if track_tangent_kernels:
        final_test_tangent_kernels = get_tangent_kernels(
            model, test_loader_det, optimizer=optimizer, 
            device=device
        )
    else:
        final_test_tangent_kernels = None
        
    if not eval_model_only:

        # Do we need the code here?

    #     init_train_loss, init_train_acc = perform_epoch(model_init, train_loader, device=device, loss_name=loss_name)
    #     init_test_loss, init_test_acc = perform_epoch(model_init, test_loader, device=device, loss_name=loss_name)

    #     init_logits = get_logits(model_init, test_loader_det, device=device)

        def get_suffix(output_inc, hidden_incs, input_inc):
            suffix = ['_']
            if output_inc:
                suffix += ['a']
            if hidden_incs:
                suffix += ['v']
            if input_inc:
                suffix += ['w']
            return ''.join(suffix)

        results_decomposition = {}
        for output_inc in [False, True]:
            for hidden_incs in [False, True]:
                for input_inc in [False, True]:
                    suffix = get_suffix(output_inc, hidden_incs, input_inc)
                    model_decomposition_term = get_model_decomposition_term(
                        model, model_init, output_inc, hidden_incs, input_inc
                    )
                    model_decomposition_term.eval()

                    final_train_loss_term, final_train_acc_term = perform_epoch(
                        lambda x: model_decomposition_term(x, model_to_get_masks_from=model), train_loader, 
                        device=device, loss_name=loss_name
                    )
                    results_decomposition['final_train_loss'+suffix] = final_train_loss_term
                    results_decomposition['final_train_acc'+suffix] = final_train_acc_term

                    final_test_loss_term, final_test_acc_term = perform_epoch(
                        lambda x: model_decomposition_term(x, model_to_get_masks_from=model), test_loader, 
                        device=device, loss_name=loss_name
                    )
                    results_decomposition['final_test_loss'+suffix] = final_test_loss_term
                    results_decomposition['final_test_acc'+suffix] = final_test_acc_term

                    final_logits_term = get_logits(
                        lambda x: model_decomposition_term(x, model_to_get_masks_from=model),
                        test_loader_det, device=device
                    )
                    final_var_f_term = np.var(final_logits_term, axis=0)[0]
                    results_decomposition['final_logits'+suffix] = final_logits_term[:100]
                    results_decomposition['final_var_f'+suffix] = final_var_f_term

        with torch.no_grad():
            input_weight_mean_abs_inc = torch.mean(
                torch.norm(model.input_layer.weight.data - model_init.input_layer.weight.data, dim=1)
            ).cpu().item()
            hidden_weight_mean_abs_inc = []
            for layer, layer_init in zip(model.hidden_layers, model_init.hidden_layers):
                if hasattr(layer, 'weight'):
                    if layer.weight is not None:
                        hidden_weight_mean_abs_inc.append(
                            torch.mean(torch.abs(layer.weight.data - layer_init.weight.data)).cpu().item()
                        )
            output_weight_mean_abs_inc = torch.mean(
                torch.abs(model.output_layer.weight.data - model_init.output_layer.weight.data)
            ).cpu().item()
    
        results = {
            #'model_state_dict': model.cpu().state_dict(), 'model_init_state_dict': model_init.cpu().state_dict(),
            'model_state_dict': None, 'model_init_state_dict': None,
            
            'training_time': training_time,
            'inference_time': inference_time,

            'train_losses': train_losses, 'train_accs': train_accs,
            'test_losses': test_losses, 'test_accs': test_accs,
            'test_logits': test_logits,
            'test_tangent_kernels': test_tangent_kernels,

            'final_train_loss': final_train_loss, 'final_train_acc': final_train_acc,
            'final_test_loss': final_test_loss, 'final_test_acc': final_test_acc,
            'final_test_logits': final_test_logits,
            'final_test_tangent_kernels': final_test_tangent_kernels,

    #         'init_train_loss': init_train_loss, 'init_train_acc': init_train_acc,
    #         'init_test_loss': init_test_loss, 'init_test_acc': init_test_acc,
    #         'init_logits': init_logits[:100],

            'input_weight_mean_abs_inc': input_weight_mean_abs_inc,
            'hidden_weight_mean_abs_inc': hidden_weight_mean_abs_inc,
            'output_weight_mean_abs_inc': output_weight_mean_abs_inc
        }

        for key in results_decomposition.keys():
            results[key] = results_decomposition[key]
            
    else:
        results = {
            #'model_state_dict': model.cpu().state_dict(), 'model_init_state_dict': model_init.cpu().state_dict(),
            'model_state_dict': None, 'model_init_state_dict': None,

            'training_time': training_time,
            'inference_time': inference_time,

            'train_losses': train_losses, 'train_accs': train_accs,
            'test_losses': test_losses, 'test_accs': test_accs,
            'test_logits': test_logits,
            'test_tangent_kernels': test_tangent_kernels,

            'final_train_loss': final_train_loss, 'final_train_acc': final_train_acc,
            'final_test_loss': final_test_loss, 'final_test_acc': final_test_acc,
            'final_test_logits': final_test_logits,
            'final_test_tangent_kernels': final_test_tangent_kernels,
        }
    
    return results