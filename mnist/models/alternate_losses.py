import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Takes two images, turns them into probability distribution,
#then takes kl divergence. Alternative choice to MSE,
#interested in how it works.
def softmax_kl(p, q):
    p = F.softmax(p).to(device)
    q = F.softmax(q).to(device)

    s1 = torch.sum(p * torch.log(p / q)) #this should be only val
    s2 = torch.sum((1-p) * torch.log((1-p)/(1-q))) #seen online, im desperate

    return s1 + s2

#Takes input to network, out_vals of format (encoded, decoded),
# and rho. Returns kl(in,out)
# + kl(middle, rho) where rho is some small activation
#Encourages sparsity.
def sparse_softmax_kl(in_vals, out_vals, hidden_dims, rho_val=0):
    rho = torch.FloatTensor([rho_val for _ in range(hidden_dims)]).unsqueeze(0).to(device)
    avg_middle_vals =  torch.sum(out_vals[0], dim=0, keepdim=True)
    #print("sparsity portion of loss: {}".format(softmax_kl(rho, avg_middle_vals)))
    
    batch_size = in_vals.size()[0]
    #acc_sparsity_loss = 0
    #for i in range(batch_size):
    #    acc_sparsity_loss += softmax_kl(rho, out_vals[0][i])
        
    return softmax_kl(in_vals, out_vals[1]) + softmax_kl(rho, avg_middle_vals)#(acc_sparsity_loss / batch_size)

def regular_softmax_kl(in_vals, out_vals, placeholder1, placeholder2=0):
    return softmax_kl(in_vals, out_vals[1])

def regular_mse(in_vals, out_vals, placeholder1, placeholder2=0):
    return nn.MSELoss()(in_vals, out_vals[1])

def sparse_mse(in_vals, out_vals, hidden_dims, rho_val=.05):
    rho = torch.FloatTensor([rho_val for _ in range(hidden_dims)]).unsqueeze(0).to(device)
    avg_middle_vals = torch.sum(out_vals[0], dim=0, keepdim=True)
    loss = nn.MSELoss()
    return loss(in_vals, out_vals[1]) + softmax_kl(rho, avg_middle_vals)

def skl_wrapper(sparsity):
    if sparsity:
        return sparse_softmax_kl
    else:
        return regular_softmax_kl

def mse_wrapper(sparsity):
    if sparsity:
        return sparse_mse
    else:
        return regular_mse

def init_loss(loss_name, sparsity):
    if loss_name == "mse":
        return mse_wrapper(sparsity)
    if loss_name == "softmax_kl":
        return skl_wrapper(sparsity)
    else:
        raise NameError("Bad loss name passed to autoencoder")
