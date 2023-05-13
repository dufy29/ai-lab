""" 超分模型,支持动态输入，实现动态放大的超分辨率模型

现在，假设我们要做一个超分辨率的应用。我们的用户希望图片的放大倍数能够自由设置。
而我们交给用户的，只有一个 .onnx 文件和运行超分辨率模型的应用程序。我们在不修改 .onnx 文件的前提下改变放大倍数。

实践环节：
    pytorch模型---> onnx  ---> onnxRuntime 加载推理

参考：
    模型部署入门教程（二）：解决模型部署中的难题
    https://zhuanlan.zhihu.com/p/479290520

模型部署系列教程涉及：
    中间表示 ONNX 的定义标准
    PyTorch 模型转换到 ONNX 模型的方法
    推理引擎 ONNX Runtime、TensorRT 的使用方法
    部署流水线 PyTorch - ONNX - ONNX Runtime/TensorRT 的示例及常见部署问题的解决方法
    MMDeploy C/C++ 推理 SDK


"""


import os

import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn
from torch.nn.functional import interpolate 




class NewInterpolate(torch.autograd.Function): 
    """
    要决定新算子映射到 ONNX 算子的方法。映射到 ONNX 的方法由一个算子的 symbolic 方法决定。
    """
    @staticmethod 
    def symbolic(g, input, scales): 
        return g.op("Resize", 
                    input, 
                    g.op("Constant", 
                         value_t=torch.tensor([], dtype=torch.float32)), 
                    scales, 
                    coordinate_transformation_mode_s="pytorch_half_pixel", 
                    cubic_coeff_a_f=-0.75, 
                    mode_s='cubic', 
                    nearest_mode_s="floor") 
 
    @staticmethod 
    def forward(ctx, input, scales): 
        scales = scales.tolist()[-2:] 
        return interpolate(input, 
                           scale_factor=scales, 
                           mode='bicubic', 
                           align_corners=False)

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        """
        SuperResolutionNet 未修改之前，nn.Upsample 在初始化阶段固化了放大倍数，
        而 PyTorch 的 interpolate 插值算子可以在运行阶段选择放大倍数。
        因此，我们在新脚本中使用 interpolate 代替 nn.Upsample，从而让模型支持动态放大倍数的超分。

        由于 PyTorch 中 interpolate 的 scale_factor 参数必须是一个数值，我们使用 torch.Tensor.item() 
            来把只有一个元素的 torch.Tensor 转换成数值
        """
        # x = interpolate(x,
        #                 scale_factor=upscale_factor.item(),
        #                 mode = 'bicubic',
        #                 align_corners=False)
        x = NewInterpolate.apply(x, upscale_factor) 
        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

# Download checkpoint and test image
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
    'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
names = ['./model/srcnn/srcnn.pth', './model/srcnn/face.png']
for url, name in zip(urls, names):
    print(f'name: {name}')
    if not os.path.exists(name):
        open(name, 'wb').write(requests.get(url).content)

def init_torch_model():
    torch_model = SuperResolutionNet()

    model_ = torch.load(names[0])
    state_dict = model_['state_dict']
    print(type(model_))
    for k in model_.keys():
        print(k)

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

model = init_torch_model()
factor = torch.tensor([1, 1, 5, 5], dtype=torch.float) 

input_img = cv2.imread(names[1]).astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img),  factor).detach().numpy()

# NCHW to HWC
# 为了让模型输出成正确的图片格式，我们把模型的输出转换成 HWC 格式，
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("./tmp/face_torch_3.png", torch_output)

# ---------------------------------------------------------------------------

print(f'='*30)
""" 记录计算图---> ONNX

"""
print(f'开始生成中间表示——ONNX....')
x = torch.randn(1, 3, 256, 256)

onnx_path = r"./model/srcnn/srcnn2.onnx"
with torch.no_grad():
    torch.onnx.export(
        model,
        (x, factor),
        onnx_path,
        opset_version=11,  # ONNX 算子集的版本
        input_names=['input', 'factor'],  # 是输入、输出 tensor 的名称
        output_names=['output'])

print(f'检查onnx 模型准确性....')
import onnx
onnx_model = onnx.load(onnx_path)
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")
    print(f"通过拖入网站{'https://netron.app'} 查看可视化结果")

# ---------------------------------------------------------------------------
print(f'='*30)
print(f'开始推理引擎——ONNX Runtime....')
print(f'ONNX模型加载路径：{onnx_path}')
import onnxruntime

input_factor = np.array([1, 1, 5, 6], dtype=np.float32) 
ort_session = onnxruntime.InferenceSession(onnx_path)
print(f'{type(input_img)}')
ort_inputs = {'input': input_img,
              'factor': input_factor}  # 注意输入输出张量的名称需要和torch.onnx.export 中设置的输入输出名对应。
ort_output = ort_session.run(None, 
                             ort_inputs)[0]

# 后处理部分
ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("tmp/face_ort_3.png", ort_output)

print(f'DONE!')