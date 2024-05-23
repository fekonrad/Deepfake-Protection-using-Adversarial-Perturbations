import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.fftpack import dct, idct
import cv2


Q_TABLE = np.array(
          [ [16,    11,    10,    16,    24,    40,    51,    61],
            [12,    12,    14,    19,    26,    58,    60,    55],
            [14,    13,    16,    24,    40,    57,    69,    56],
            [14,    17,    22,    29,    51,    87,    80,    62],
            [18,    22,    37,    56,    68,   109,   103,    77],
            [24,    35,    55,    64,    81,   104,   113,    92],
            [49,    64,    78,    87,   103,   121,   120,   101],
            [72,    92,    95,    98,   112,   100,   103,    99] ] )


def quantization_compress(img, quality_factor=50, patch_size=8):
    """
    :param img:
    :return:
    """
    assert img.shape[0] % patch_size == 0 and img.shape[1] % patch_size == 0

    dct_ = compute_dct(img, patch_size)
    q_matrix = compute_q_table(quality_factor)

    # divide dct by q_table, round to int, multiply by q_table again
    num_patches_x = int(img.shape[1] / patch_size)
    num_patches_y = int(img.shape[0] / patch_size)

    for p_x in range(num_patches_x):
        for p_y in range(num_patches_y):
            patch = dct_[p_y*patch_size : (p_y+1)*patch_size, p_x*patch_size : (p_x+1)*patch_size]
            patch = np.divide(patch, q_matrix)
            patch = np.round(patch).astype(int)
            dct_[p_y*patch_size : (p_y+1)*patch_size, p_x*patch_size : (p_x+1)*patch_size] = np.multiply(patch, q_matrix)

    # return inverse DCT of matrix
    return compute_dct_inv(dct_)


def compute_dct(img, patch_size=8):
    dct_ = np.zeros((img.shape[0], img.shape[1]))
    num_patches_x = int( img.shape[1] / patch_size )
    num_patches_y = int( img.shape[0] / patch_size )

    for p_x in range(num_patches_x):
        for p_y in range(num_patches_y):
            patch = img[p_y*patch_size : (p_y+1)*patch_size, p_x*patch_size : (p_x+1)*patch_size]
            dct_patch = dct(dct(patch, norm='ortho').T, norm='ortho').T
            dct_[p_y*patch_size : (p_y+1)*patch_size, p_x*patch_size : (p_x+1)*patch_size] = dct_patch

    return dct_


def compute_dct_inv(dct, patch_size=8):
    dct_inv = np.zeros((dct.shape[0], dct.shape[1]))
    num_patches_x = int( dct.shape[1] / patch_size )
    num_patches_y = int( dct.shape[0] / patch_size )

    for p_x in range(num_patches_x):
        for p_y in range(num_patches_y):
            patch = dct[p_y * patch_size: (p_y + 1) * patch_size, p_x * patch_size: (p_x + 1) * patch_size]
            dct_inv_patch = idct(idct(patch, norm='ortho').T, norm='ortho').T
            dct_inv[p_y * patch_size: (p_y + 1) * patch_size, p_x * patch_size: (p_x + 1) * patch_size] = dct_inv_patch

    return dct_inv


def compute_q_table(quality_factor):
    assert 1 <= quality_factor <= 100
    if quality_factor < 50:
        s = 5000 / quality_factor
    else:
        s = 200 - 2 * quality_factor
    table = np.floor((Q_TABLE * s + 50) / 100)
    table[table == 0] = 1           # avoid division by 0
    return table


if __name__ == "__main__":
    """
    img = Image.open("data/cat.jpg")
    img = np.array(img)
    img = img[10:730]


    dec_img = np.zeros((720, 1200, 3))
    for c in range(3):
        channel = img[:, :, c]
        dec_img[:, :, c] = quantization_compress(channel, quality_factor=5)
    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(dec_img.astype(int))
    ax[1].axis('off')
    ax[2].imshow(np.abs(img - dec_img).astype(int))
    ax[2].axis('off')
    plt.show()
    """

    rand_img = np.random.randint(low=0, high=255, size=(64, 64))
    dec_img = quantization_compress(rand_img, quality_factor=30)
    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(rand_img)
    ax[0].axis("off")
    ax[1].imshow(dec_img.astype(int))
    ax[1].axis("off")
    ax[2].imshow(np.abs(rand_img - dec_img).astype(int))
    ax[2].axis("off")
    plt.show()

