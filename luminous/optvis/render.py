import torch
import torchvision
import numpy as np

from luminous.optvis import transform, objectives

_layer_output = None
_hook = None

def setup_hook(layer):
    global _hook
    if _hook is not None:
        _hook.remove()

    def hook(module, input, output):
        global _layer_output
        _layer_output = output

    _hook = layer.register_forward_hook(hook)

class no_ReLU:
    def __enter__(self):
        self.old_relu = torch.nn.functional.relu
        torch.nn.functional.relu = lambda x, inplace=False: x
    def __exit__(self, type, value, traceback):
        torch.nn.functional.relu = self.old_relu

def remove_ReLU(layer):
    old_forward = layer.forward
    def forward(x):
        with no_ReLU():
            return old_forward(x)
    layer.forward = forward

def create_vis_tensor(size, batch_n=1, mean=0.5, std=0.05, device=None):
    tensor = torch.randn(batch_n, 3, size, size, device=device) * std + mean
    return tensor

def visualize(param, net, offset, objective=None,
              transforms=(lambda x: x), thresholds=(256,),
              progress=True, render=True, optimizer=None, device=None, lr=0.05):
    if progress:
        from tqdm import tqdm_notebook as tqdm
    if render:
        from IPython.display import display

    if optimizer is None:
        optimizer = torch.optim.Adam([param.tensor], lr=lr)
    if objective is None:
        objective = objectives.channel()

    net.train(False)
    param.tensor.requires_grad_(True)

    images = []

    iters = max(thresholds)
    iterable = range(iters)
    if progress:
        iterable = tqdm(iterable)

    for i in iterable:

        optimizer.zero_grad()
        net(transforms(param.to_valid_rgb()))
        loss = objective(offset, _layer_output, device=device)
        loss.backward()
        optimizer.step()

        grad_rms = (param.tensor.grad**2).mean().sqrt()

        if i+1 in thresholds:
            vis = param_to_images(param)
            images.append(vis)
            print(i+1, grad_rms, loss)
            if render:
                display(vis)

    return images

def param_to_images(param):
    rgb = param.to_valid_rgb().cpu()
    combined_tensor = torch.cat(list(rgb), dim=2)
    return torchvision.transforms.ToPILImage()(combined_tensor)

