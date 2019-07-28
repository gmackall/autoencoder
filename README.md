Repository with simple example of autoencoder and classifier on MNIST, in pytorch.
Requirements can be installed using conda environment file in this directory.
Plan is to make a more complicated text autoencoder with combined reconstruction loss + loss of text quality at bottleneck.

## Getting started with MNIST portion
There are many tutorials online of how to use pytorch, so this isn't really one of those. This is just a collection of simple and segmented code for personal use/reference that could serve as a useful base for others. Models are defined in mnist/models/, currently there is just a super basic autoencoder and MLP. The autoencoder could be improved in many ways, the most obvious of which is to make it a regularized autoencoder, discussed [here](https://www.deeplearningbook.org/contents/autoencoders.html) in 14.2.