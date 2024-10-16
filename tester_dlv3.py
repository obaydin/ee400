# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 08:25:01 2024

@author: STAJYER
"""
import numpy as np
import torch
from modeling.deeplab import *
import scipy.io

model_load_name=r"C:\Users\STAJYER\Desktop\models\pytorch-deeplab-xception-master\trained_models\son_durak.pth"
# data_path=r"C:\Users\STAJYER\Desktop\data_staj\snr_20\pixs_5_5\specs\snr_20pixes5_5_v6.mat"
data_path=r"C:\Users\STAJYER\Desktop\reel_data\data2.mat"
device= torch.device('cpu')

model = DeepLab(num_classes=2,
                backbone='resnet',
                output_stride=16,
                )
# cp = torch.load(r"C:\Users\STAJYER\Desktop\models\pytorch-deeplab-xception-master\run\pascal\deeplab-resnet\checkpoint.pth.tar")

# model.load_state_dict(cp['state_dict'])
checkpoint = torch.load(model_load_name, map_location=device)
state_dict = {k.replace('module.',''): v for k,v in checkpoint.items() }
model.load_state_dict(state_dict)
# inputt=np.load(data_path)
a=scipy.io.loadmat(data_path)
inputt=a['parca']
inputt = (inputt - np.mean(inputt))/np.std(inputt)
model.eval()
image = torch.from_numpy(inputt).unsqueeze_(0).float()
image.unsqueeze(0)
image=image.unsqueeze_(0)
inputs=image.to(device)

with torch.no_grad():
    preds=model(inputs)
_,output = torch.max(preds,1)

output_np=output.numpy()
s={"pred":output_np,"input":inputt}

scipy.io.savemat("result.mat",s)

