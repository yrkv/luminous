import torch
import numpy as np

class Param():
    """
    expected to define self.tensor as the leaf
    """

    def to_valid_rgb(self, decorrelate=False, sigmoid=True):
        raise NotImplementedError


# ported from Lucid
color_correlation_svd_sqrt = torch.tensor([
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
])
# ported from Lucid
max_norm_svd_sqrt = float(
    np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
)
color_mean = 0.45

# ported from Lucid
def _linear_decorelate_color(rgb):
    """Multiply input by sqrt of empirical (ImageNet) color correlation matrix.
    If you interpret t's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations.
    """

    batch, ch, h, w = rgb.shape
    t_flat = rgb.permute(0,2,3, 1).reshape(-1, 3)

    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    t_flat = torch.matmul(t_flat, color_correlation_normalized.T)

    t = t_flat.view(batch, h, w, 3).permute(0, 3, 1, 2)
    return t

class Image(Param):

    def __init__(self, size, batch_n=1, sd=0.01, decorrelate=False, sigmoid=True, device='cuda'):
        self.tensor = torch.randn(batch_n, 3, size, size, device=device) * sd
        self.device = device
        self.decorrelate = decorrelate
        self.sigmoid = sigmoid

    # ported from Lucid
    def to_valid_rgb(self):
        t = self.tensor
        if self.decorrelate:
            t = _linear_decorelate_color(t)
        if self.decorrelate and not self.sigmoid:
            t += self.color_mean
        if self.sigmoid:
            t = torch.sigmoid(t)
        else:
            t = torch.clamp(t, 0, 1)
        return t

class FFT(Param):

    def __init__(self, size, batch_n=1, sd=0.01, decorrelate=False,
                 sigmoid=True, device='cuda'):
        self.shape = (batch_n, 3, size, size)
        self.tensor, self.freqs = FFT.fft_param(self.shape, sd, device)
        self.device = device
        self.decorrelate = decorrelate
        self.sigmoid = sigmoid

    # ported from Lucid
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

    # ported from Lucid
    def fft_param(shape, sd, device):
        """An image paramaterization using 2D Fourier coefficients."""

        batch, ch, h, w = shape
        freqs = FFT.rfft2d_freqs(h, w)
        init_val_size = (batch, ch) + freqs.shape + (2,)

        init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
        
        spectrum_t = torch.from_numpy(init_val)
        return spectrum_t.to(device), torch.from_numpy(freqs, device=device)

    # ported from Lucid
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
        return image_t.float()

    # ported from Lucid
    def to_valid_rgb(self):
        t = FFT.fft_param_to_image(self.shape, self.tensor, self.freqs)
        if self.decorrelate:
            t = _linear_decorelate_color(t)
        if self.decorrelate and not self.sigmoid:
            t += self.color_mean
        if self.sigmoid:
            t = torch.sigmoid(t)
        else:
            t = torch.clamp(t, 0, 1)
        return t

