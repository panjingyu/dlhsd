import torch


def find_z(net, inputs, targets, criterion):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = net.eval()(inputs)
    loss_z = criterion(outputs, targets)
    loss_z.backward()
    grad = inputs.grad
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach()
    eps = 1e-7
    z = (z + eps) / (z.flatten(start_dim=1).norm(dim=1) + eps)
    inputs.grad.zero_()
    net.zero_grad()

    return z.detach(), norm_grad


def regularizer(net, inputs, targets, criterion, h=3., lambda_=4, device='cuda'):
    '''
    Regularizer term in CURE
    '''
    z, norm_grad = find_z(net, inputs, targets, criterion)
    bs = inputs.size(0)

    inputs.requires_grad_()
    outputs_pos = net.eval()(inputs + h * z)
    outputs_orig = net.eval()(inputs)

    loss_pos = criterion(outputs_pos, targets)
    loss_orig = criterion(outputs_orig, targets)
    grad_diff = torch.autograd.grad(loss_pos - loss_orig, inputs,
                                    create_graph=True)[0]
    reg = grad_diff.reshape(bs, -1).norm(dim=1)
    reg = reg * reg

    return lambda_ * reg.sum() / bs, norm_grad
