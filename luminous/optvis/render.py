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

# taken directly from lucid
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)

# taken fro lucid and modified to work with pytorch
def fft_param(shape, sd=None):
    """An image paramaterization using 2D Fourier coefficients."""

    sd = sd or 0.01
    batch, ch, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, ch) + freqs.shape + (2,)

    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    
    spectrum_t = torch.from_numpy(init_val)
    return spectrum_t, torch.from_numpy(freqs)

def fft_param_to_image(shape, spectrum_t, freqs, decay_power=1):
    
    batch, ch, h, w = shape
    
    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar leanring rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)
    scale = scale[None, None, :, :, None]
    scaled_spectrum_t = scale * spectrum_t

    # convert complex scaled spectrum to shape (ch, h, w) image tensor
    image_t = torch.irfft(scaled_spectrum_t, 2)

    # in case of odd spatial input dimensions we need to crop
    image_t = image_t[:batch, :ch, :h, :w]
    image_t = image_t / 4.0  # TODO: is that a magic constant?
    return image_t

def visualize(tensor, net, offset, objective=None,
              transforms=(lambda x: x), thresholds=(256,), blur=0.,
              progress=True, render=True, optimizer=None, device=None,
              pre_render=None):
    if progress:
        from tqdm import tqdm_notebook as tqdm
    if render:
        from IPython.display import display

    if optimizer is None:
        optimizer = torch.optim.Adam([tensor], lr=0.05)
    if objective is None:
        objective = objectives.channel()
    if pre_render is None:
        pre_render = lambda x: x

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
            vis = tensor_to_images(pre_render(tensor).sigmoid())
            images.append(vis)
            print(i+1, grad_rms)
            if render:
                display(vis)

    return images

def tensor_to_images(tensor):
    combined_tensor = torch.cat(list(tensor.cpu()), dim=2)
    return torchvision.transforms.ToPILImage()(combined_tensor)

