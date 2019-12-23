import torch
import torchvision
import numpy as np

from luminous.optvis import transform, objectives

_layer_output = None

def setup_hook(layer):
    def hook(module, input, output):
        global _layer_output
        _layer_output = output

    layer.register_forward_hook(hook)

class no_ReLU:
    def __enter__(self):
        self.old_relu = F.relu
        F.relu = lambda x, inplace=False: x
    def __exit__(self, type, value, traceback):
        F.relu = self.old_relu

def remove_ReLU(layer):
    old_forward = layer.forward
    def forward(x):
        with no_ReLU():
            return old_forward(x)
    layer.forward = forward

def create_vis_tensor(size, batch_n=1, mean=0.5, std=0.05, device=None):
    tensor = torch.randn(batch_n, 3, size, size, device=device) * std + mean
    return tensor

def visualize(tensor, net, offset, objective=None,
              transforms=(lambda x: x), thresholds=(256,), blur=0.,
              progress=True, render=True, optimizer=None, device=None):
    if progress:
        from tqdm import tqdm_notebook as tqdm
    if render:
        from IPython.display import display

    if optimizer is None:
        optimizer = torch.optim.Adam([tensor], lr=0.05)
    if objective is None:
        objective = objectives.channel()

    if '__iter__' not in dir(blur):
        blur = (blur, blur)

    net.train(False)
    tensor.requires_grad_(True)

    images = []

    iters = max(thresholds)
    iterable = range(iters)
    if progress:
        iterable = tqdm(iterable)

    for i in iterable:
        sigma = blur[0] + ((blur[1] - blur[0]) * i) / iters

        optimizer.zero_grad()
        net(transforms(tensor))
        loss = objective(offset, _layer_output, device=device)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if sigma > 0:
                tensor.set_(transform.blur(tensor, sigma))

        grad_rms = (tensor.grad**2).mean().sqrt()

        if i+1 in thresholds:
            vis = tensor_to_images(tensor.sigmoid())
            images.append(vis)
            print(i+1, grad_rms)
            if render:
                display(vis)

    return images

def tensor_to_images(tensor):
    combined_tensor = torch.cat(list(tensor.cpu()), dim=2)
    return torchvision.transforms.ToPILImage()(combined_tensor)

