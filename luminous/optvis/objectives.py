import torch

def channel():
    def inner(offset, layer_output, device):
        batch_n = len(layer_output)
        activations = layer_output[:, offset:offset+batch_n]
        I = torch.eye(batch_n, device=device).view(batch_n, batch_n, 1, 1)
        loss = -(activations * I).mean()
        return loss
    return inner

def neuron():
    def inner(offset, layer_output, device):
        batch_n = len(layer_output)
        middle = layer_output.shape[-1] // 2
        activations = layer_output[:, offset:offset+batch_n, middle, middle]
        I = torch.eye(batch_n, device=device)
        loss = -(activations * I).mean()
        return loss
    return inner

def deepdream():
    def inner(offset, layer_output, device):
        loss = -layer_output.mean()
        return loss
    return inner
