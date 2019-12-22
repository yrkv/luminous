import torch

def channel(offset):
    def inner(layer_output, device=None):
        batch_n = len(layer_output)
        activations = layer_output[:, offset:offset+batch_n]
        I = torch.eye(batch_n, device=device).view(batch_n, batch_n, 1, 1)
        loss = -(activations * I).mean()
        return loss
    return inner

def neuron(offset):
    def inner(layer_output, device=None):
        batch_n = len(layer_output)
        middle = activations.shape[-1] // 2
        activations = layer_output[:, offset:offset+batch_n, middle, middle]
        I = torch.eye(batch_n, device=device)
        loss = -(activations * I).mean()
        return loss
    return inner
