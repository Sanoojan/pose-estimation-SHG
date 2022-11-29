import argparse
import os.path
from PIL import Image
import numpy as np
from torch.nn.functional import mse_loss, kl_div, softmax
import cv2 as cv
import sys
sys.path.append("src")

import torch
import torch.backends.cudnn
from torch.nn import DataParallel

from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.mpii import Mpii, print_mpii_validation_accuracy
from stacked_hourglass.utils import imutils as imu
from stacked_hourglass.predictor import HumanPosePredictor

import colorsys

def HSVToRGB(h, s, v):
 (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
 return (int(255*r), int(255*g), int(255*b))

def getDistinctColors(n):
 huePartition = 1.0 / (n + 1)
 colors = [HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)]
 return colors


def get_model(arch, model_file,):
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    pretrained = not model_file

    if pretrained:
        print('No model weights file specified, using pretrained weights instead.')

    # Create the model, downloading pretrained weights if necessary.
    if arch == 'hg1':
        model = hg1(pretrained=pretrained)
    elif arch == 'hg2':
        model = hg2(pretrained=pretrained)
    elif arch == 'hg8':
        model = hg8(pretrained=pretrained)
    else:
        raise Exception('unrecognised model architecture: ' + model)
    model = model.to(device)

    if not pretrained:
        assert os.path.isfile(model_file)
        print('Loading model weights from file: {}'.format(model_file))
        checkpoint = torch.load(model_file)
        state_dict = checkpoint['state_dict']
        if sorted(state_dict.keys())[0].startswith('module.'):
            model = DataParallel(model)

        if 0: ## EDIT: Load weights of 2hg to 1hg
            state_dict = get_flexible_weights(model, state_dict)

        model.load_state_dict(state_dict)

    return model



if __name__ == '__main__':
    IMAGEPATH  = "data/images"
    WEIGHTPATH = "checkpoint/hg2-base/checkpoint.pth.tar"
    SAVEPATH = "checkpoint/checkup3/"
    colors = getDistinctColors(16)
    select_images=200
    # Load the image.
    List_all=os.listdir(IMAGEPATH)
    selected_images=List_all[:select_images]
    # print(selected_images)

    model = get_model('hg2', WEIGHTPATH)

    for im in selected_images:
        # image = Image.open(os.path.join(IMAGEPATH,im))
        
        img = imu.load_image(os.path.join(IMAGEPATH,im))
        poseObj = HumanPosePredictor(model)
        heatmap = poseObj.estimate_heatmaps(img)

        
        # print(heatmap.max(), heatmap.min())
        # heatmap=torch.clamp(heatmap, 0, 1)
        # heatmap=heatmap/torch.max(heatmap)
        # heatmap=softmax(heatmap,dim=0)
        
        # heatmap = np.power(heatmap, 3.0)
        # heatmap=heatmap/torch.max(heatmap)
        print(heatmap.max(), heatmap.min())
        heatmap=heatmap.cpu().numpy()
        points = poseObj.estimate_joints(img)
        #load image in opencv
        npim = cv.imread(os.path.join(IMAGEPATH,im))

        heatplot = np.zeros((3, 64,64))
        for n in range(heatmap.shape[0]):
            clr = colors[n]
            heatplot[0] += heatmap[n]*clr[0]
            heatplot[1] += heatmap[n]*clr[1]
            heatplot[2] += heatmap[n]*clr[2]
            pt  = points[n]
            cv.circle(npim, (int(pt[0]), int(pt[1])), 15, clr, -1)
            cv.circle(npim, (int(pt[0]), int(pt[1])), 15, (0,0,0), 3)


        heatplot = np.clip(heatplot, 0, 255).transpose(1,2,0).astype(np.uint8)
        # resize with opencv
        heatplot = cv.resize(heatplot, (npim.shape[1], npim.shape[0]))
        heatplot = np.asarray(heatplot, dtype=float)

        npim = npim * 0.7 + heatplot * 0.7
        npim = np.clip(npim, 0, 255).astype(np.uint8)

        cv.imwrite(os.path.join(SAVEPATH,im), npim)

    # heatmap=heatmap/np.amax(heatmap,keepdims=True)
    # attention=np.array(attention).reshape(224,224,1)
    # gamma=0.7
    # hetmp=(255.0*(np.power(attention, gamma))).astype(np.uint8)
    # # hetmp=(255.0*np.array(attention).reshape(224,224,1)).astype(np.uint8)
    # hetmp = cv.blur(hetmp,(10,10))
    # attn=cv.applyColorMap(hetmp,cv.COLORMAP_JET)
    # image=torchvision.transforms.ToPILImage()(image) 
    # # image=einops.rearrange(image,'c h w -> h w c')
    # # print(np.array(attn).shape)
    # # print(np.array(image).shape)
    # # print(attn.dtype)
    # # print(image.dtype)
    # # print(f_name_ori)
    # # try:
    # resu=cv.addWeighted(np.array(image),0.7,np.array(attn),0.6,0.4)
    # img_orig=cv.addWeighted(np.array(image),1.0,np.array(attn),0.0,0.0)
    # cv.imwrite(fname,resu)
    # cv.imwrite(f_name_ori,img_orig)

