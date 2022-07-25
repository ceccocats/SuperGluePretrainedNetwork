import os
from ssl import match_hostname
from xml.dom import minicompat
from numpy import int32
import torch

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
torch.set_grad_enabled(False)

from models.utils import *
import matplotlib.cm as cm
import json
import math
import matplotlib.pyplot as plt
import matplotlib


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \"{}\"'.format(device))

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.8,
    }
}
superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)

image0, inp0, scales0 = read_image("/home/cecco/ws/rtls/tk.legacy/build/GPIO_run3/keyframes/1650532384030129920_1.png", device, [848, 800], 0, False)
data = inp0

# plt.imshow(inp0.cpu().reshape(848,800))
# plt.show()

# test
output = superpoint(data)
print(output.shape)

plt.imshow(output.cpu().reshape(800,848))
plt.show()

# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(superpoint, data)

# retest
output = traced_script_module(data)
#print(output.shape)

plt.imshow(output.cpu().reshape(800,848))
plt.show()

traced_script_module.save("traced_superpoint2.pt")