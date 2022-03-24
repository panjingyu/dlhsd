import torch
from torch.autograd.gradcheck import zero_gradients


def find_z(net, inputs, targets, criterion, h, device='cuda'):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = net.eval()(inputs)
    loss_z = criterion(net.eval()(inputs), targets)
    loss_z.backward(torch.ones(targets.size()).to(device))
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
    zero_gradients(inputs)
    net.zero_grad()

    return z, norm_grad


def regularizer(net, inputs, targets, criterion, h=3., lambda_=4, device='cuda'):
    '''
    Regularizer term in CURE
    '''
    z, norm_grad = find_z(inputs, targets, h)

    inputs.requires_grad_()
    outputs_pos = net.eval()(inputs + z)
    outputs_orig = net.eval()(inputs)

    loss_pos = criterion(outputs_pos, targets)
    loss_orig = criterion(outputs_orig, targets)
    grad_diff = torch.autograd.grad((loss_pos-loss_orig),
                                    inputs,
                                    grad_outputs=torch.ones(targets.size()).to(device),
                                    create_graph=True)
    grad_diff = grad_diff[0]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    net.zero_grad()

    return torch.sum(lambda_ * reg) / float(inputs.size(0)), norm_grad