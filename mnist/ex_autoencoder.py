from models.autoencoder import AutoEncoder
from utils.plotting import auto_encoder_gen_ex
from utils.training import train

#Instantiate network using model defined in models/autoencoder.py
print("Instantiating network")
net = AutoEncoder(sparsity=True, loss="softmax_kl", bottleneck_factor=2)

#Train the network using the training defined in utils/training.py
#use 10 epochs, learning rate .01, momentum .9, batch size 32
print("Starting Training")
train(net,
      epochs=10,
      learning_rate=.01,
      momentum=.9,
      batch_size=32,
      data_root='./datasets/data/',
      save_dir='./chkpts/temp.pt')

#Generate examples
in_name = "testins.png"
out_name = "testouts.png"
print("Training done, generating example images. Input images will be saved in ./{} and reconstructed (output) images will be saved in ./{}".format(in_name, out_name))
auto_encoder_gen_ex(net,
                    in_name=in_name,
                    out_name=out_name,
                    columns = 4,
                    batch_size = 32,
                    data_root="./datasets/data/")
print("Done!")
