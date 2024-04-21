import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from utils import visualize_cam_3channel, visualize_cam_1channel, visualize_all_feature_maps
from Gradcam import GradCAM

def gradcam_result(alexnet, img_path, output_path='./output_grad/'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_name = os.path.basename(img_path).split('.')[0]
    img = PIL.Image.open(img_path)

    img_arr = np.array(img)
    img_arr.setflags(write=1)

    torch_img = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    normed_torch_img = torch_img

    # 获取预测结果
    pred = alexnet(normed_torch_img)
    pred = F.softmax(pred, dim=1)
    pred = pred.squeeze().cpu().detach().numpy()
    pred_idx = np.argmax(pred)
    print('Predicted:', pred_idx)

    alexnet_model_dict = dict(arch=alexnet, layer_name='features_12', input_size=(224, 224))
    alexnet_gradcam = GradCAM(alexnet_model_dict)

    # 根据预测结果获取GradCAM结果
    mask1, _, activations = alexnet_gradcam(normed_torch_img, class_idx=pred_idx)
    # 根据反预测结果获取GradCAM结果
    mask2, _, activations = alexnet_gradcam(normed_torch_img, class_idx=1 - pred_idx)

    visualize_all_feature_maps(activations, img_name)

    heatmap1, result1 = visualize_cam_3channel(mask1, torch_img)
    heatmap2, result2 = visualize_cam_3channel(mask2, torch_img)

    heatmapa1, heatmapb1, heatmapc1 = visualize_cam_1channel(mask1, torch_img)
    heatmapa2, heatmapb2, heatmapc2 = visualize_cam_1channel(mask2, torch_img)

    grid_image1 = make_grid(torch.stack([torch_img.squeeze().cpu(), heatmap1, result1]), nrow=3)
    grid_image2 = make_grid(torch.stack([torch_img.squeeze().cpu(), heatmap2, result2]), nrow=3)


    grid_image3 = make_grid(torch.stack([torch_img.squeeze().cpu()[0].unsqueeze(0), heatmapa1, torch_img.squeeze().cpu()[1].unsqueeze(0), heatmapb1, torch_img.squeeze().cpu()[2].unsqueeze(0), heatmapc1]), nrow=2)
    grid_image4 = make_grid(torch.stack([torch_img.squeeze().cpu()[0].unsqueeze(0), heatmapa2, torch_img.squeeze().cpu()[1].unsqueeze(0), heatmapb2, torch_img.squeeze().cpu()[2].unsqueeze(0), heatmapc2]), nrow=2)

    save_image(grid_image1, output_path + img_name + '_result1.jpg')
    save_image(grid_image2, output_path + img_name + '_result2.jpg')
    save_image(grid_image3, output_path + img_name + '_result3.jpg')
    save_image(grid_image4, output_path + img_name + '_result4.jpg')

if __name__ == '__main__':
    alexnet = torch.load('./model/torch_alex.pth')
    print(alexnet)
    alexnet.eval().cuda()
    
    # 遍历data4文件夹下的所有图片
    img_path = './data4/'
    for i in os.listdir(img_path):
        tmp_path = img_path + i
        print(tmp_path)
        gradcam_result(alexnet, tmp_path)
    