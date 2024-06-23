import torch
from torch import nn
import torch.nn.functional as F


class SDIMLoss(nn.Module):
    """
        Implements the Structural (Dis-) Similarity Index Measure to be used as a loss function.
        see: https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    """
    def __init__(self, window_size=16):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        """
            assumes x and y to be of common shape (b, c, h, w)
        """
        window = torch.ones((1, x.shape[1], self.window_size, self.window_size)) / (self.window_size ** 2)
        conv = lambda img: F.conv2d(img,
                                   weight=window,
                                   stride=(self.window_size, self.window_size),
                                   padding=0
                                  )
        mean_x, mean_y = conv(x), conv(y)
        mean_x_sq, mean_y_sq = mean_x ** 2, mean_y ** 2
        var_x, var_y = conv(x**2) - mean_x_sq, conv(y**2) - mean_y_sq

        cov_xy = conv(x * y) - mean_x * mean_y

        c1, c2 = (0.01 * 255.0)**2, (0.03 * 255.0)**2           # see Wikipedia for justification of these constants

        numerator = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
        denominator = (mean_x_sq + mean_y_sq + c1) * (var_x + var_y + c2)

        return (1 - torch.div(numerator, denominator).mean(dim=(1, 2, 3))) / 2.0


class MultiScaleSDIMLoss(nn.Module):
    """
        Implements the Multiscale Structural (Dis-) Similarity Index Measure to be used as a loss function.
        see: https://www.cns.nyu.edu/pub/eero/wang03b.pdf
        (this implementation differs slightly to the method described in this paper!)
    """
    def __init__(self, window_sizes=[8, 16, 32, 64]):
        super().__init__()
        self.window_sizes = window_sizes
        self.sdims = nn.ModuleList([SDIMLoss(window_size=window_size) for window_size in window_sizes])

    def forward(self, x, y):
        loss = 0.0
        for loss_fn in self.sdims:
            loss += loss_fn(x, y)

        return loss
