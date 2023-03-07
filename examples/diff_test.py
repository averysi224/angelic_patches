import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import pdb

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

# im = torch.ones(1,3,512,512)*255
dtype = torch.FloatTensor
im = Variable((torch.ones(1,3,512,512)*255).type(dtype), requires_grad=True)

#Test:
# dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#im should be a 4D tensor of shape B x C x H x W with type dtype, range [0,255]:
# plt.imshow(im.squeeze(0).permute(1,2,0)/255) #To plot it im should be 1 x C x H x W

# plt.figure()
#Rotation by np.pi/2 with autograd support:
rotated_im = rot_img(im, np.pi/4, dtype) # Rotate image by 90 degrees.
plt.imshow(rotated_im.squeeze(0).permute(1,2,0).detach().numpy()/255)

plt.show()