from models.autoencoder import AutoEncoder
from utils.plotting import auto_encoder_gen_ex
from utils.training import train

#Instantiate network using model defined in models/autoencoder.py
print("Instantiating network")
net = AutoEncoder()

#Train the network using the training defined in utils/training.py
#use 10 epochs, learning rate .01, momentum .9, batch size 32
print("Starting Training")
train(net, 10, .01, .9, 32, './datasets/data/')

#Generate examples
print("Training done, generating example images. Input images will be saved in ./ins.png and reconstructed (output) images will be saved in ./outs.png")
auto_encoder_gen_ex(net,
                    in_name="ins.png",
                    out_name="outs.png",
                    columns = 8,
                    batch_size = 32,
                    data_root="./datasets/data/")
print("Done!")
