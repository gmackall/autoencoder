from matplotlib import pyplot as plt
import numpy as np
import torch

def tiled(inputs, out_name, columns = 1):
    n_images = inputs.shape[0]
    fig = plt.figure()
    for i in range(n_images):
        a = fig.add_subplot(columns, np.ceil(n_images/float(columns)), i + 1)
        plt.imshow(inputs[i][0])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig(out_name)
