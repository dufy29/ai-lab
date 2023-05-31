import torch
import torch.nn as nn
import torch.nn.parallel as par
import torch.optim as optim

# initialize torch . distributed properly
# with init_process_group

# setup model and optimizer
net = nn.Linear (10 , 10)
net = par.DistributedDataParallel(net)
opt = optim.SGD(net.parameters() , lr =0.01)

# run forward pass
inp = torch . randn (20 , 10)
exp = torch . randn (20 , 10)
out = net(inp)

# run backward pass
nn.MSELoss()(out , exp).backward()

# update parameters
opt.step()