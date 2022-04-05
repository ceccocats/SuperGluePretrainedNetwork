import torch
import urllib
from PIL import Image
from torchvision import transforms
import numpy as np
import struct 
import os
from torchsummary import summary
import torch.nn as nn
from torch.jit import trace

def create_folders():
    if not os.path.exists('debug'):
        os.makedirs('debug')
    if not os.path.exists('layers'):
        os.makedirs('layers')

def bin_write(f, data):
    data =data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def hook(module, input, output):
    setattr(module, "_value_hook", output)

def exp_input(model, input_batch):
    # Export the input batch 
    i = input_batch["image"].cpu().data.numpy()
    i = np.array(i, dtype=np.float32)
    i.tofile("debug/input.bin", format="f")
    print("input: ", i.shape)

def print_wb_output(model):
    f = None
    for n, m in model.named_modules():
        if not(' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type)):
            continue

        #print(m.type)

        in_output = m._value_hook
        o = in_output.cpu().data.numpy()
        o = np.array(o, dtype=np.float32)
        t = '-'.join(n.split('.'))
        o.tofile("debug/" + t + ".bin", format="f")
        print('------- ', n, ' ------') 
        print("debug  ",o.shape)
        
        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type):
            file_name = "layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')

        w = np.array([])
        b = np.array([])
        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            w = m._parameters['weight'].cpu().data.numpy()
            w = np.array(w, dtype=np.float32)
            print ("    weights shape:", np.shape(w))
        
        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            print ("    bias shape:", np.shape(b))
        
        if 'BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].cpu().data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.cpu().data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.cpu().data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f,b)
            bin_write(f,s)
            bin_write(f,rm)
            bin_write(f,rv)
            print ("    b shape:", np.shape(b))
            print ("    s shape:", np.shape(s))
            print ("    rm shape:", np.shape(rm))
            print ("    rv shape:", np.shape(rv))

        else:
            bin_write(f,w)
            if b.size > 0:
                bin_write(f,b)

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            f.close()
            print("close file")
            f = None


def export_to_tkDNN(model, input_batch, net_name):
    # create folders debug and layers if do not exist
    create_folders()

    # add output attribute to the layers
    for n, m in model.named_modules():
        m.register_forward_hook(hook)

    model.eval()
    out = None
    with torch.no_grad():
        out = model(input_batch)

    # export input bin
    exp_input(model, input_batch)

    # export layers weights and outputs
    print_wb_output(model)

    with open(net_name+".txt", 'w') as f:
        for item in list(model.children()):
            f.write("%s\n" % item)
