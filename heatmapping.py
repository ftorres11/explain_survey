# -*- coding: utf-8 -*-
import imageio as io
import numpy as np
import shutil 
import skimage as ski
import pdb
import os
import cv2

osp = os.path
osj = osp.join

path_saliency = osj('SaliencyMaps', 'ImageNet', 'ResNet50')
path_images = osj('..', '..', 'ILSVRC_2012', 'val')
path_data = osj('data', '200classes_5inst.csv')
path_storage = osj('Heatmapped', 'ImageNet', 'ResNet50')

list_options = os.listdir(path_saliency)

with open(path_data, 'r') as data:
    data_list = data.readlines()

if not osp.exists(osj(path_storage, 'Original')):
    os.makedirs(osj(path_storage, 'Original'))

imgs = []
for line in data_list:
    img_file, label = line.strip().split(',')
    shutil.copyfile(osj(path_images, img_file),
                    osj(path_storage, 'Original', img_file))
    imgs.append(img_file)

for salient in list_options:
    path_sal = osj(path_storage, salient, 'groundtruth')
    path_src = osj(path_saliency, salient, 'groundtruth')
    if not osp.exists(path_sal):
        os.makedirs(path_sal)
    for instance in imgs:
        try:
            smap = np.load(osj(path_src, instance.replace('.JPEG', '.npy')))
            orig = io.imread(osj(path_images, instance))
            
            if len(smap.shape)>3:
                smap = smap[0]
                smap = np.moveaxis(smap, 0, -1)
            if smap.shape[-1] == 1:
                smap = smap[:,:,0]
            smap = ski.transform.resize(smap, (orig.shape[0], orig.shape[1]))
            smap = 1-(smap*255).astype(np.uint8)
            if len(orig.shape) < 3:
                orig = cv2.merge([orig, orig, orig])
            #pdb.set_trace()
            cam_mask = cv2.merge([smap, smap, smap])
            heatmap = cv2.applyColorMap(smap, cv2.COLORMAP_JET)
            heatmap = heatmap.astype(float)/255
            overlap = heatmap + orig.astype(float)/255
            overlap = overlap / np.max(overlap)
            io.imwrite(osj(path_sal, instance), (overlap*255).astype(np.uint8))
        except ValueError:
            #if len(orig.shape) < 3 and len(smap.shape) > 2:
            #    smap = np.mean(smap, axis=-1)
            #else:
            #    smap = np.repeat(smap[:, :, np.newaxis], 3, axis=2)
            #overlap = ((orig.astype(float)*smap.astype(float))/255).astype(np.uint8)
            #io.imwrite(osj(path_sal, instance), overlap)
            continue
