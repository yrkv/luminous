import torch
import kornia

def roll(images, d):
    dx, dy = (torch.randn(2) * d).round().int()
    return images.roll(shifts=(dy, dx), dims=(2, 3))

def blur(images, sigma, size=11):
    return kornia.gaussian_blur2d(images, (size, size), (sigma, sigma))

def random_crop(t, d=4):
    batch_n = t.shape[0]
    size = t.shape[-1]
    
    boxes = torch.rand(batch_n, 4, 2) * d
    c = torch.tensor([
                      [False, False],
                      [False, True],
                      [True, False],
                      [True, True],
    ])
    boxes[:, c] = size-1 - boxes[:, c]
    t = kornia.crop_and_resize(t, boxes, (size, size))
    return t
