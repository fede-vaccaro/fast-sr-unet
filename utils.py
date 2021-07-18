import argparse
import os
from pathlib import Path


class ARArgs:

    def __init__(self, args=None):
        ap = argparse.ArgumentParser()
        archs = ['srunet', 'unet', 'espcn', 'srresnet']
        ap.add_argument("-ds", "--dataset", type=str, default="",
                        help="Dir from where to import the datasets")
        ap.add_argument("-m", "--model", type=str, default=None,
                        help="path to *specific* model checkpoint to load")
        ap.add_argument("-dv", "--device", type=str, default="0",
                        help="CUDA device to be used. For info type '$ nvidia-smi'")
        ap.add_argument("-v", "--verbose", action='store_true', default=False,
                        help="Verbosity mode.")
        ap.add_argument("--vidpatches", type=str, default="",
                        help="Where to store/load video patches.")
        ap.add_argument("--export", type=str, default="",
                        help="Where to export models.")
        ap.add_argument("-e", "--epochs", type=int, default=80,
                        help="Number of epochs you want to train the model.")
        ap.add_argument("--clipname", type=str, default="",
                        help="[RENDER.PY ONLY] path to the clip you want to upscale")
        ap.add_argument("--arch", type=str, default="srunet", choices=archs,
                        help="Which network architecture to train.")
        ap.add_argument("--w0", type=float, default=1.0,
                        help="LPIPS weight")
        ap.add_argument("--w1", type=float, default=1.0,
                        help="SSIM Weight")
        ap.add_argument("--l0", type=float, default=0.001,
                        help="Adversarial Component Weight")
        ap.add_argument("--upscale", type=int, default=2,
                        help="Default upscale factor, obbtained as resolution ratio between LQ and HQ samples")
        ap.add_argument("--layer_mult", type=float, default=1.0, help="Layer multiplier - SR UNet only")
        ap.add_argument("--n_filters", type=int, default=64, help="Net Number of filters param - SR UNet and UNet only")
        ap.add_argument("--downsample", type=float, default=1.0, help="Downsample factor, SR Unet and UNet only")
        ap.add_argument("--testdir", type=str, default="test",
                        help="[TEST ONLY] Where the test clips are contained.")
        ap.add_argument("--testinputres", type=int, default=540, help="[TEST ONLY] Input testing resolution")
        ap.add_argument("--testoutputres", type=int, default=1080, help="[TEST ONLY] Output testing resolution")
        ap.add_argument("--crf", type=int, default=23, help="Reference compression CRF")
        ap.add_argument('--show-only-upscaled', dest='show-only-upscaled', action='store_true',
                        help="[RENDER.PY ONLY] If you want to show only the neural net upscaled version of the video")

        if args is None:
            args = vars(ap.parse_args())
        else:
            args = vars(ap.parse_args(args))

        self.MODEL_NAME = args['model']
        self.VERBOSE = args['verbose']
        self.DATASET_DIR = Path(args['dataset'])
        self.CUDA_DEVICE = args['device']
        self.VID_PATCHES = args['vidpatches']
        self.EXPORT_DIR = args['export']
        self.N_EPOCHS = int(args['epochs'])
        self.CLIPNAME = args['clipname']
        self.ARCHITECTURE = args['arch']
        self.VALIDATION_FREQ = 1
        self.W0 = args['w0']
        self.W1 = args['w1']
        self.L0 = args['l0']
        self.UPSCALE_FACTOR = args['upscale']
        self.LAYER_MULTIPLIER = args['layer_mult']
        self.N_FILTERS = args['n_filters']
        self.DOWNSAMPLE = args['downsample']
        self.TEST_INPUT_RES = args['testinputres']
        self.TEST_OUTPUT_RES = args['testoutputres']
        self.CRF = args['crf']
        self.TEST_DIR = args['testdir']
        self.SHOW_ONLY_HQ = args['show-only-upscaled']

        self.archs = archs


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def show_tensor(t):
    import data_loader

    img_tensor = t[0]
    img = data_loader.de_transform(img_tensor)
    img.show()


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    import torch
    import math
    from torch import nn as nn

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
