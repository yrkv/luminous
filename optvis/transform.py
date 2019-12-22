import torch
import kornia

def roll(images, d):
    dx, dy = (torch.randn(2) * d).round().int()
    return images.roll(shifts=(dy, dx), dims=(2, 3))

def blur(images, sigma, size=11):
    return kornia.gaussian_blur2d(images, (size, size), (sigma, sigma))
