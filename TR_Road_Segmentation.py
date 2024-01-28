__author__ = 'mkv-aql'

import numpy as np
import fastai
from fastai.vision.all import *
#from fastai.vision.interpret import *
from fastai.callback.all import *
#from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.test_utils import *
#from fastai.utils.mem import *
import torch
torch.backends.cudnn.benchmark=True
import os
import PIL

def open_image(fname, size=224):
    img = PIL.Image.open(fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

#Data Prepartaion
path = Path('C:/Users/AGAM MUHAJIR/Desktop/Thiago_Rateke_Dataset/')
files = os.listdir(path)

print(files)
#path.ls()

#codes = np.loadtxt(path/'codes.txt', dtype = str); codes
codes = np.loadtxt(path/'codes.txt', dtype = str)
print(codes)

#Define Images and Labels
path_lbl = path/'RTK_SemanticSegmentationGT_NoColorMapMasks'
path_img = path/'RTK_SemanticSegmentationGT_originalFrames'

fnames = get_image_files(path_img)
print(fnames[:3])
print(len(fnames))

lbl_names = get_image_files(path_lbl)
print(lbl_names[:3])
print(len(lbl_names))

img_f = fnames[139]
#img = open_image(img_f) #fastai version 1
img = load_image(img_f) #fastai version 2
#img.show(figsize=(5,5)) #fastai version 1

#Showing images
#open_image(files[0]).shape
print(open_image(img_f).shape) #See tensor information
#print(img)
im = PIL.Image.open(img_f) #See image with PIL
im.show()
'''
plt.imshow(im) #See image with plt
plt.show()
'''

#Infere mask filename
get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

#mask = open_mask(get_y_fn(img_f)) #Fastai Version 1
mask = OpenMask(get_y_fn(img_f)) #Fastai version 2
mask.show(figsize=(5,5), alpha=1)

mask_array = np.array(mask)
src_size = np.array(mask.shape[1:])
#src_size,mask.data
print(src_size)
#print(mask_array)