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
import struct

import sys

def bin_write(f, data):
    data =data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \"{}\"'.format(device))

config = {
    'superpoint': {
        'nms_radius': 3,
        'keypoint_threshold': 0.001,
        'max_keypoints': 2048, #1024,
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.8,
    }
}
superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
superglue = SuperGlue(config.get('superglue', {})).eval().to(device)


input_dir = sys.argv[1]
print(input_dir)
files = sorted( [ i for i in os.listdir(input_dir) if i.endswith("png") ] )
files_json = sorted( [ i for i in os.listdir(input_dir) if i.endswith("json") ] )
assert len(files) == len(files_json)

features = {}

poses2d = []

for i in range(len(files)):
 
    f_json = open(input_dir + "/" + files_json[i])
    pose_json = json.load(f_json)
    poses2d.append((pose_json["pose"]["px"], pose_json["pose"]["py"]))

    imageFile = files[i]
    #print("extract: ", imageFile)
    print("extract: ", i, "of", len(files))

    image0, inp0, scales0 = read_image(input_dir + "/" + imageFile, device, [848, 800], 0, False)

    data = {'image': inp0}
    pred = superpoint(data)
    data = {**data, **pred}
    data = {k: [i.cpu() for i in v] for k, v in data.items()}
    

    kpts = data["keypoints"][0].numpy()    
    scores = data["scores"][0].numpy()
    desc = data["descriptors"][0].numpy() 

    print(np.shape(desc))
    print(np.shape(desc[:,0]))
    print(desc[:,0])

    out = input_dir + "/" + files[i] + ".feats"
    print(out)

    f = open(out, "wb")
    bin_write(f, np.array(np.shape(kpts), dtype=np.float32))
    bin_write(f, kpts)
    bin_write(f, np.array(np.shape(scores), dtype=np.float32))
    bin_write(f, scores)
    bin_write(f, np.array(np.shape(desc), dtype=np.float32))
    bin_write(f, desc)
    f.close()

    

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

        
