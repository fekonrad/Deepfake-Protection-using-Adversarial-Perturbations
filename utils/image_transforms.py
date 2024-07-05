import torch


def rgb_to_ycbcr(rgb_tensor):
    """
    Converts an RGB tensor to YCbCr.

    :param rgb_tensor: (torch.Tensor) Input tensor of shape (b, c, h, w) in RGB format.
    :returns: (torch.Tensor) Output tensor of shape (b, c, h, w) in YCbCr format.
    """
    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.168736, -0.331264, 0.5],
                                     [0.5, -0.418688, -0.081312]], dtype=torch.float32)
    shift = torch.tensor([0, 128, 128], dtype=torch.float32).view(1, 3, 1, 1)

    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)
    ycbcr_tensor = torch.matmul(rgb_tensor, transform_matrix.T)
    ycbcr_tensor = ycbcr_tensor.permute(0, 3, 1, 2) + shift
    return ycbcr_tensor


def ycbcr_to_rgb(ycbcr_tensor):
    """
    Converts a YCbCr tensor to RGB.

    :param ycbcr_tensor: (torch.Tensor) Input tensor of shape (b, c, h, w) in YCbCr format.
    :returns: (torch.Tensor) Output tensor of shape (b, c, h, w) in RGB format.
    """
    transform_matrix = torch.tensor([[1.0, 0.0, 1.402],
                                     [1.0, -0.344136, -0.714136],
                                     [1.0, 1.772, 0.0]], dtype=torch.float32)
    shift = torch.tensor([0, 128, 128], dtype=torch.float32).view(1, 3, 1, 1)

    ycbcr_tensor = ycbcr_tensor - shift

    ycbcr_tensor = ycbcr_tensor.permute(0, 2, 3, 1)
    rgb_tensor = torch.matmul(ycbcr_tensor, transform_matrix.T)
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
    return rgb_tensor


def denormalize(tensor, mean, std):
    """
        Inverts the normalization torchvision.transforms.Normalize(mean, std)
    :param tensor: torch.tensor of shape (b, c, h, w) representing normalized images
    :param mean: array of length `c` representing the mean of each color channel
    :param std: array of length `c` representing the standard deviation of each color channel
    :return: torch.tensor of shape (b, c, h, w) representing the original image, before normalization
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor
