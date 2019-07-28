from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import math

from .data_loader import load_data

#could just use from torchvision.utils import make_grid

def tiled(inputs, out_name, columns = 1):
    n_images = inputs.shape[0]
    fig = plt.figure()
    for i in range(n_images):
        a = fig.add_subplot(columns, np.ceil(n_images/float(columns)), i + 1)
        plt.imshow(inputs[i][0])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig(out_name)

def auto_encoder_gen_ex(network, in_name='ins.png', out_name='outs.png', columns=1, data_root='../datasets/data', batch_size=32, small_edge=800):
    loader = load_data(batch_size, data_root)
    (x, _) = next(iter(loader))
    tiled(x, in_name, columns)
    i_dms = network.image_dims
    tiled(network(x).detach().cpu().view(batch_size, i_dms[0], i_dms[1], i_dms[2]).numpy(), out_name, columns)

    rows = math.ceil(batch_size/columns)
    if rows > columns:
        width = small_edge
        height = int(small_edge * rows/columns)
    else:
        height = small_edge
        width = int(small_edge * columns/rows)
    reduce_dimensions(in_name, width, height)
    reduce_dimensions(out_name, width, height)

def reduce_dimensions(im_file, width, height):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(im_file)
    im = im.resize((width, height), Image.ANTIALIAS)
    im.save(im_file)
