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
        #a = fig.add_subplot(columns, np.ceil(n_images/float(columns)), i + 1)
        a = fig.add_subplot(np.ceil(n_images/float(columns)), columns, i + 1)
        plt.imshow(inputs[i][0])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig(out_name)

def auto_encoder_gen_ex(network, in_name='ins.png', out_name='outs.png', columns=1, data_root='../datasets/data', batch_size=32, small_edge=800):
    loader = load_data(batch_size, data_root)
    (x, _) = next(iter(loader))
    tiled(x, in_name, columns)
    i_dms = network.image_dims
    tiled(network(x)[1].detach().cpu().view(batch_size, i_dms[0], i_dms[1], i_dms[2]).numpy(), out_name, columns)

    reduce_dimensions(in_name, 800)
    reduce_dimensions(out_name, 800)

def reduce_dimensions(im_file, small_edge):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(im_file)
    x, y = im.size
    if x > y:
        new_y = small_edge
        new_x = int(small_edge * (x / y))
    else:
        new_x = small_edge
        new_y = int(small_edge * (y / x))
    im = im.resize((new_x, new_y), Image.ANTIALIAS)
    im.save(im_file)
