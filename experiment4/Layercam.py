import torch
import torch.nn.functional as F
from utils import find_alexnet_layer

class LayerCAM(object):
    def __init__(self, model_dict):
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = find_alexnet_layer(self.model_arch, layer_name)
        # 获取目标层的前向输出和反向梯度
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        try:
            input_size = model_dict['input_size']
            device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
            self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
        except KeyError:
            print("无法正确实现前向传播")
            pass

    def forward(self, input, class_idx=None, retain_graph=False):

        b, c, h, w = input.shape

        logit = self.model_arch(input)

        # 根据类别索引获取预测结果
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()

        # 分数反向回传
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']

        # b, k, u, v = gradients.size()

        activation_maps = (activations * F.relu(gradients))
        saliency_map = activation_maps.sum(1, keepdim=True)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
         # 归一化到0-1之间
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit, activations

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
