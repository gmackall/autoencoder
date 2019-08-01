Repository with simple example of autoencoder and classifier on MNIST, in pytorch.
Requirements can be installed using conda environment file in this directory.
Plan is to make a more complicated text autoencoder with combined reconstruction loss + loss of text quality at bottleneck.

## Getting started with MNIST portion
The first thing to do to get started would be to install conda, and then load and activate the environment stored in requirements.yml. Once that is done you should be ready to go.

There are many tutorials online of how to use pytorch, so this isn't really one of those. This is just a collection of simple and segmented code for personal use/reference that could serve as a useful base for others. Models are defined in mnist/models/, currently there is just a super basic autoencoder and MLP. The autoencoder could be improved in many ways, the most obvious of which is to make it a regularized autoencoder, discussed [here](https://www.deeplearningbook.org/contents/autoencoders.html) in 14.2. I have added a sparsity penalization inspired by Andrew Ng's [CS294A lecture notes.](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) The obvious next step would be to add more layers.

For a sample usage of this code, see mnist/ex_autoencoder.py. Looking at the code there should help show how the pieces fit together. Running it trains a very simple autoencoder (unregularized) which has an architecture of input layer (width 784) -> hidden layer (width half that of input) -> output layer (width 784). It trains for 10 epochs, and then outputs sample input images and their reconstructions.

For example, a sample input could be the images:

![alt text](https://github.com/gmackall/autoencoder/blob/master/mnist/ins.png "Inputs")

which then generate a sample output:

![alt text](https://github.com/gmackall/autoencoder/blob/master/mnist/outs.png "Reconstructed inputs")

By adding regularization in the form of penalizing KL-divergence between activations in the hidden layer and some small value rho, say rho=.05, you can achieve better reconstruction as seen below (different inputs but you get the idea):

![alt text](https://github.com/gmackall/autoencoder/blob/master/mnist/testouts.png "With regularization")

To really show it is a better reconstruction it would be better to compare MSE or some unbiased metric, but I would say it pretty clearly looks better and you can do that yourself if you are interested (let me know if it isn't). One can enable or disable sparsity by a flag passed to the network at instantiation, look to mnist/ex_autoencoder.py for examples, or look at the init for the network itself. Suggested uses for this repository are as a guide to base your code off of, or to use for feature extraction for training another network.