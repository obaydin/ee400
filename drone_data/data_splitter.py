


import os
import scipy.io
import numpy as np
import math
import random
import glob
from tqdm import tqdm



matching_folders=[]
mat_files=[]
root_directory=r"C:\Users\STAJYER\Desktop\data_easy"

target_name='binary'

for foldername, subfolders, filenames in os.walk(root_directory):
    if target_name in subfolders:
        matching_folders.append(os.path.join(foldername,target_name))
        
original_list=[]
for m in range(len(matching_folders)):
    temp= glob.glob(os.path.join(matching_folders[m],'**\\*.mat'),recursive=True)
    original_list.append(temp)
    
flattened_list=[]
for sublist in original_list:
    for item in sublist:
        flattened_list.append(item)
        
for l in tqdm(range(len(flattened_list))):
    bnr_direc= flattened_list[l]
    
    #get the spec direcs
    spec_direc=bnr_direc.replace("binary", "specs")
    spec_direc= spec_direc[:-7]
    spec_direc =spec_direc +".mat"
    
    
    mat=scipy.io.loadmat(spec_direc)
    spectrogram_full=mat['P']
    mat_bnr= scipy.io.loadmat(bnr_direc)
    label_mat=mat_bnr['p']
    label_mat = np.array(label_mat,dtype=bool)
    time_len=125
    freq_len=800
    M=freq_len
    N=time_len
    
    freq_part= math.floor(spectrogram_full.shape[0]/freq_len)
    time_part= math.floor(spectrogram_full.shape[1]/time_len)
    spectrogram_full=spectrogram_full[0:freq_part*freq_len,0:time_part*time_len]
    label_mat= label_mat[0:freq_part*freq_len,0:time_part*time_len]
    tiles=[spectrogram_full[x:x+M,y:y+N] for x in range(0,spectrogram_full.shape[0],M) for y in range(0,spectrogram_full.shape[1],N)]
    tiles_bnr=[label_mat[x:x+M,y:y+N] for x in range(0,label_mat.shape[0],M) for y in range(0,label_mat.shape[1],N)]
    
    
    for m in range(len(tiles)):
        spectrogram= tiles[m]
        label= tiles_bnr[m]
        
        #binary save path
        save_direc_bnr=bnr_direc.replace("data_easy", "data_easy_faster")
        bnr_npy_direc=save_direc_bnr[:-7]+"_"+str(m)+"bnr.npy"
        
        #spec save path
        save_direc_spec=spec_direc.replace("data_easy", "data_easy_faster")
        spec_npy_direc=save_direc_spec[:-4]+"_"+str(m)+".npy"
        
        
        spec_write_dir=os.path.dirname(spec_npy_direc)
        if os.path.isdir(spec_write_dir)==False:
            os.makedirs(spec_write_dir)
            
        bnr_write_dir=os.path.dirname(bnr_npy_direc)
        if os.path.isdir(bnr_write_dir)==False:
            os.makedirs(bnr_write_dir)
            
        #save data
        np.save(spec_npy_direc,np.array(spectrogram))
        np.save(bnr_npy_direc,np.array(label))
    
    
    
    
    
    
    