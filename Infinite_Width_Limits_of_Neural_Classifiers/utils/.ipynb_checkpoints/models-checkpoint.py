from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def iterate_layers(model):
    children = list(model.children())
    if len(children) == 0:
        yield model
    else:
        for child in children:
            for grandchild in iterate_layers(child):
                yield grandchild
                
                
class LinearizedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super(LinearizedModel, self).__init__()
        
        self.model = model
        self.model_init = deepcopy(model)
        self.model_init_for_grad = deepcopy(model)
        
        self.input_layer = self.model.input_layer
        self.hidden_layers = self.model.hidden_layers
        self.output_layer = self.model.output_layer
        
    def forward(self, X):
        return self.model(X).detach() - self.model_init(X).detach() + self.model_init_for_grad(X)
    
    
class OptimizerForLinearizedModel():
    def __init__(self, optimizer: optim.Optimizer, linearized_model: LinearizedModel):
        super(OptimizerForLinearizedModel, self).__init__()
        
        self.linearized_model = linearized_model
        
        self.optimizer = optimizer
        
        self.params = list(self.linearized_model.model.parameters())
        self.params_init_for_grad = list(self.linearized_model.model_init_for_grad.parameters())
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.linearized_model.model_init_for_grad.zero_grad()
        
    def step(self):
        for param, param_init_for_grad in zip(self.params, self.params_init_for_grad):
            param.grad = param_init_for_grad.grad
        
        self.optimizer.step()


class CustomBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input, use_running_mean=False):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or not self.track_running_stats or not use_running_mean:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, torch.mean(input, dim=0), self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
    

def get_normalization_layer(width, normalization, dim):
    if normalization == 'none':
        layer = nn.Identity()
        
    elif normalization.startswith('batch'):
        if normalization == 'batch':
            affine = False
        elif normalization == 'batch_affine':
            affine = True
        else:
            raise ValueError
            
        if dim == 1:
            layer = CustomBatchNorm1d(width, affine=affine)
        elif dim == 2:
            layer = nn.BatchNorm2d(width, affine=affine)
        elif dim == 3:
            layer = nn.BatchNorm3d(width, affine=affine)
        else:
            raise ValueError
        
    elif normalization.startswith('layer'):
        if normalization == 'layer':
            affine = False
        elif normalization == 'layer_affine':
            affine = True
        else:
            raise ValueError
            
        layer = nn.LayerNorm(width, elementwise_affine=affine)
        
    else:
        raise ValueError
        
    return layer


def _get_negative_slope(activation):
    if activation == 'lrelu':
        return 0.1
    elif activation == 'relu':
        return 0.
    else:
        raise ValueError


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU
    elif activation == 'lrelu':
        return lambda: nn.LeakyReLU(negative_slope=_get_negative_slope(activation))
    elif activation == 'softplus':
        return nn.Softplus
    elif activation == 'elu':
        return nn.ELU
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError
        
        
class FCNet(nn.Module):
    def __init__(self, input_shape, num_classes, width, 
                 num_hidden=1, bias=True, normalization='none', activation='relu'):
        super(FCNet, self).__init__()
        
        self.activation = activation
        
        activation = get_activation(activation)
        
        input_size = int(np.prod(input_shape))
        
        self.input_layer = nn.Linear(input_size, width, bias=bias)
        
        hidden_layers = []
        hidden_layers.append(activation())
        hidden_layers.append(get_normalization_layer(width, normalization, dim=1))
        
        for _ in range(num_hidden-1):
            hidden_layers.append(nn.Linear(width, width, bias=bias))
            hidden_layers.append(activation())
            hidden_layers.append(get_normalization_layer(width, normalization, dim=1))
            
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.output_layer = nn.Linear(width, num_classes, bias=bias)
        
    def forward(self, X, model_to_get_masks_from=None):
        if model_to_get_masks_from is None:
            Z = self.input_layer(X.view(X.shape[0], -1))
            Z = self.hidden_layers(Z)
            return self.output_layer(Z.view(Z.shape[0], -1))
        
        elif isinstance(model_to_get_masks_from, FCNet):
            assert self.activation == model_to_get_masks_from.activation
            assert self.activation in ['relu', 'lrelu']
            assert len(self.hidden_layers) == len(model_to_get_masks_from.hidden_layers)
            
            Z = self.input_layer(X.view(X.shape[0], -1))
            Z_for_mask = model_to_get_masks_from.input_layer(X.view(X.shape[0], -1))
            for layer, layer_for_mask in zip(self.hidden_layers, model_to_get_masks_from.hidden_layers):
                if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                    negative_slope = _get_negative_slope(self.activation)
                    Z = Z * ((Z_for_mask > 0).float() + negative_slope * (Z_for_mask < 0).float())
                    Z_for_mask = F.leaky_relu(Z_for_mask, negative_slope=negative_slope)
                elif isinstance(layer, CustomBatchNorm1d):
                    Z = layer(Z, use_running_mean=True)
                    Z_for_mask = layer_for_mask(Z_for_mask)
                else:
                    Z = layer(Z)
                    Z_for_mask = layer_for_mask(Z_for_mask)
            return self.output_layer(Z.view(Z.shape[0], -1))
        else:
            raise ValueError
    
    
def make_conv_block(input_shape: tuple, num_convs: int, num_kernels, kernel_size, padding: int, activation, final_pool,
                    bias: bool, use_bn: bool, bn_affine: bool, dropout_rate: float):
    out_channels, h, w = input_shape
    activation = get_activation(activation)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_convs
    if isinstance(num_kernels, int):
        num_kernels = [num_kernels] * num_convs
    
    layers = []
    for k_size, num_k in zip(kernel_size, num_kernels):
        in_channels, out_channels = out_channels, num_k
        layers += [nn.Conv2d(in_channels, out_channels, k_size, padding=padding, bias=bias), activation()]
        h = h - (k_size - 1) + padding * 2
        w = w - (k_size - 1) + padding * 2
        if use_bn:
            layers += [nn.BatchNorm2d(out_channels, affine=bn_affine)]
    
    if final_pool == 'max':
        layers += [nn.MaxPool2d(2)]
        h = h // 2
        w = w // 2
    elif final_pool == 'global_avg':
        layers += [nn.AvgPool2d((h,w))]
        h = 1
        w = 1
    else:
        raise ValueError

    layers += [nn.Dropout2d(dropout_rate)]

    output_shape = (out_channels, h, w)
                
    return layers, output_shape

    
# ------------- ConvLarge ---------------

    
class ConvLarge(nn.Module):
    def __init__(self, input_shape, num_classes, init_num_kernels, 
                 num_conv_blocks, num_convs_per_block, num_convs_in_final_block,
                 hidden_bias=False, output_bias=True, use_bn=True, bn_affine=True, dropout_rate=0.5, activation='lrelu'):
        super(ConvLarge, self).__init__()
        
        layers = []

        output_shape = input_shape
        num_kernels = init_num_kernels
        for _ in range(num_conv_blocks):
            layers_block, output_shape = make_conv_block(
                output_shape, num_convs_per_block, num_kernels, 3, 1, 
                activation, 'max', hidden_bias, use_bn, bn_affine, dropout_rate
            )
            num_kernels *= 2
            layers += layers_block

        layers_block, output_shape = make_conv_block(
            output_shape, num_convs_in_final_block, 
            list(num_kernels // (2 ** np.arange(num_convs_in_final_block))), [3] + [1] * (num_convs_in_final_block-1), 0, 
            activation, 'global_avg', hidden_bias, use_bn, bn_affine, 0
        )
        layers += layers_block
    
        layers += [nn.Flatten(), nn.Linear(output_shape[0], num_classes, bias=output_bias)]
        
        self.input_layer = layers[0]
        self.hidden_layers = nn.Sequential(*layers[1:-1])
        self.output_layer = layers[-1]
        
    def forward(self, X, model_to_get_masks_from=None): # last argument is ignored
        Z = self.input_layer(X)
        Z = self.hidden_layers(Z)
        return self.output_layer(Z)

    
# ------------- ConvSmall ---------------

    
class ConvSmall(nn.Module):
    def __init__(self, input_shape, num_classes, init_num_kernels, 
                 num_conv_blocks, num_convs_per_block, num_convs_in_final_block,
                 hidden_bias=False, output_bias=True, use_bn=True, bn_affine=True, dropout_rate=0.5, activation='lrelu'):
        super(ConvSmall, self).__init__()
        
        layers = []

        output_shape = input_shape
        num_kernels = init_num_kernels
        for _ in range(num_conv_blocks):
            layers_block, output_shape = make_conv_block(
                output_shape, num_convs_per_block, num_kernels, 3, 1, 
                activation, 'max', hidden_bias, use_bn, bn_affine, dropout_rate
            )
            num_kernels *= 2
            layers += layers_block

        layers_block, output_shape = make_conv_block(
            output_shape, num_convs_in_final_block, 
            num_kernels // 2, [3] + [1] * (num_convs_in_final_block-1), 0, 
            activation, 'global_avg', hidden_bias, use_bn, bn_affine, 0
        )
        layers += layers_block
    
        layers += [nn.Flatten(), nn.Linear(output_shape[0], num_classes, bias=output_bias)]
        
        self.input_layer = layers[0]
        self.hidden_layers = nn.Sequential(*layers[1:-1])
        self.output_layer = layers[-1]
        
    def forward(self, X, model_to_get_masks_from=None): # last argument is ignored
        Z = self.input_layer(X)
        Z = self.hidden_layers(Z)
        return self.output_layer(Z)
    
    
# ------------- ConvReg ---------------

    
class ConvReg(nn.Module):
    def __init__(self, input_shape, num_classes, init_num_kernels, 
                 num_conv_blocks, num_convs_per_block,
                 hidden_bias=False, output_bias=True, use_bn=True, bn_affine=True, dropout_rate=0.5, activation='lrelu'):
        super(ConvReg, self).__init__()
        
        layers = []

        output_shape = input_shape
        num_kernels = init_num_kernels
        for _ in range(num_conv_blocks):
            layers_block, output_shape = make_conv_block(
                output_shape, num_convs_per_block, num_kernels, 3, 1, 
                activation, 'max', hidden_bias, use_bn, bn_affine, dropout_rate
            )
            num_kernels *= 2
            layers += layers_block

        layers_block, output_shape = make_conv_block(
            output_shape, num_convs_per_block, num_kernels // 2, 3, 1, 
            activation, 'global_avg', hidden_bias, use_bn, bn_affine, 0
        )
        layers += layers_block
    
        layers += [nn.Flatten(), nn.Linear(output_shape[0], num_classes, bias=output_bias)]
        
        self.input_layer = layers[0]
        self.hidden_layers = nn.Sequential(*layers[1:-1])
        self.output_layer = layers[-1]
        
    def forward(self, X, model_to_get_masks_from=None): # last argument is ignored
        Z = self.input_layer(X)
        Z = self.hidden_layers(Z)
        return self.output_layer(Z)
    
    
# ------------- ResNet ---------------

    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BatchNorm2dNoAffine(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(nn.BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class ResNet(nn.Module):

    def __init__(self, input_shape, num_classes, width_per_group, layers, 
                 hidden_bias=False, output_bias=True, use_bn=True, bn_affine=True,
                 block=BasicBlock, zero_init_residual=False,
                 groups=1,  replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        if use_bn:
            if bn_affine:
                norm_layer = nn.BatchNorm2d
            else:
                norm_layer = BatchNorm2dNoAffine
        else:
            norm_layer = nn.Identity
        self._norm_layer = norm_layer

        self.inplanes = width_per_group
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_shape[0], self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=hidden_bias)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, width_per_group, layers[0])
        self.layer2 = self._make_layer(block, width_per_group*2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, width_per_group*4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, width_per_group*8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width_per_group*8 * block.expansion, num_classes, bias=output_bias)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            raise NotImplementedError
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

        layers = list(iterate_layers(self))
        self.input_layer = layers[0]
        assert self.input_layer is self.conv1
        self.hidden_layers = nn.Sequential(*layers[1:-1])
        self.output_layer = layers[-1]
        assert self.output_layer is self.fc

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    