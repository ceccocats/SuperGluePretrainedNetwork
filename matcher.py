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

from export import *

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
superglue = SuperGlue(config.get('superglue', {})).eval().to(device)

input_dir = "assets/freiburg_sequence/"
files = sorted( [ i for i in os.listdir(input_dir) if i.endswith("png") ] )

features = {}
for i in range(len(files)):
 
    imageFile = files[i]
    #print("extract: ", imageFile)
    print("extract: ", i, "of", len(files))

    image0, inp0, scales0 = read_image(input_dir + "/" + imageFile, device, [848, 800], 0, False)

    data = {'image': inp0}

    if(i==0):
        export_to_tkDNN(superpoint, data, "superpoint")
    
    pred = superpoint(data)
    data = {**data, **pred}
    data = {k: [i.cpu() for i in v] for k, v in data.items()}
    features[imageFile] = data


# # match all
# maxDist = 20.0
# W = 10
# for i in range(len(files)):
#     print("match: ", i, "of", len(files))

#     checked = 0
#     for j in range(len(files)): #range(i-W,i+W):
#         if i == j or j < 0 or j >= len(files):
#             continue

#         if(abs(i-j) > W):
#             dist = math.sqrt( (poses2d[i][0] - poses2d[j][0])**2 + (poses2d[i][1] - poses2d[j][1])**2 )
#             if(dist > maxDist):
#                 continue

#         img0 = files[i]
#         img1 = files[j]
#         #print("match: ", img0, img1)

#         data = {**data, **{k+'0': [i.to(device) for i in v] for k, v in features[img0].items()}}
#         data = {**data, **{k+'1': [i.to(device) for i in v] for k, v in features[img1].items()}}

#         for k in data:
#             if isinstance(data[k], (list, tuple)):
#                 data[k] = torch.stack(data[k])

#         pred = superglue(data)

#         pred = {**data, **pred}
#         pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
#         kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
#         matches, conf = pred['matches0'], pred['matching_scores0']
#         matches1, conf1 = pred['matches1'], pred['matching_scores1']
        
#         # Keep the matching keypoints.
#         valid = matches > -1
#         mkpts0 = kpts0[valid]
#         mkpts1 = kpts1[matches[valid]]
#         mconf = conf[valid]

#         kidx_0 = np.linspace(0,len(valid)-1,len(valid), dtype=int32)[valid]
#         kidx_1 = matches[valid]
        
#         n = len(kidx_0)
#         match = np.concatenate((kidx_0.reshape(n,1),kidx_1.reshape(n,1)),axis=1)
#         #print(match)
#         #print(np.shape(match))
#         np.savetxt(input_dir + '/matches/' + files[i] + "_" + files[j] + ".txt", match, delimiter=' ', fmt="%d")

#         # image0, inp0, scales0 = read_image(input_dir + "/" + img0, device, [848, 800], 0, False)
#         # image1, inp1, scales1 = read_image(input_dir + "/" + img1, device, [848, 800], 0, False)
#         # color = cm.jet(mconf)
#         # make_matching_plot(
#         #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
#         #     "", "", True, False, True, 'Matches', "")
#         # plt.show()

#         checked += 1
#     print("checked: ", checked)

        