import cv2
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
from torchvision import models

def find_alexnet_layer(arch, target_layer_name):
    # 获取目标层
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer

def visualize_cam_3channel(mask, img):
    # 可视化三通道的GradCAM结果
    mask = mask.cpu().detach().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)

    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result

def visualize_cam_1channel(mask, img):
    # 可视化单通道的GradCAM结果
    mask = mask.cpu().detach().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)

    return r, g, b


def visualize_all_feature_maps(feature_maps, save_name):
    # 可视化卷积层所有通道
    num_channels = feature_maps.shape[1]
    rows = cols = int(np.sqrt(num_channels))
    fig, axs = plt.subplots(rows, cols, figsize=(16, 16), tight_layout=True)
    for i in range(num_channels):
        ax = axs[i // cols, i % cols]
        im = ax.imshow(feature_maps[0, i].detach().cpu(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    # 创建一个新的画布，保存颜色标尺
    fig_cbar, ax_cbar = plt.subplots(figsize=(1, 16))
    cbar = fig.colorbar(im, cax=ax_cbar)
    fig_cbar.savefig(f'./img/color_scale.png')
    plt.close(fig_cbar)
    
    plt.savefig('./img/' + save_name + '_out.png')

    concatenate_images_horizontally('./img/' + save_name + '_out.png', './img/color_scale.png', './img/' + save_name + '_out.png')

def concatenate_images_horizontally(image1_path, image2_path, output_path):
    # 打开图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    width1, height1 = image1.size
    width2, height2 = image2.size

    # 确保两张图片的高度相同
    if height1 != height2:
        raise ValueError("两张图片没办法拼接。")
    new_width = width1 + width2

    concatenated_image = Image.new('RGB', (new_width, height1))
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (width1, 0))
    concatenated_image.save(output_path)






