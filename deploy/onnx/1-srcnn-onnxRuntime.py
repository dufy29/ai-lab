""" 超分模型
在这份代码中，我们创建了一个经典的超分辨率网络 SRCNN。
SRCNN 先把图像上采样到对应分辨率，再用 3 个卷积层处理图像。
为了方便起见，我们跳过训练网络的步骤，直接下载模型权重
（由于 MMEditing 中 SRCNN 的权重结构和我们定义的模型不太一样，
我们修改了权重字典的 key 来适配我们定义的模型），同时下载好输入图片。
为了让模型输出成正确的图片格式，我们把模型的输出转换成 HWC 格式，
并保证每一通道的颜色值都在 0~255 之间。
如果脚本正常运行的话，一幅超分辨率的人脸照片会保存在“face_torch.png”中。

实践环节：
    pytorch模型---> onnx  ---> onnxRuntime 加载推理
参考：
    模型部署入门教程（一）：模型部署简介, https://zhuanlan.zhihu.com/p/477743341

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

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',  # 双三次插值
            align_corners=False)

        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
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
    torch_model = SuperResolutionNet(upscale_factor=3)

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
input_img = cv2.imread(names[1]).astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW to HWC
# 为了让模型输出成正确的图片格式，我们把模型的输出转换成 HWC 格式，
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("./deploy/face_torch.png", torch_output)

# ---------------------------------------------------------------------------

print(f'='*30)
""" 记录计算图---> ONNX

深度学习模型实际上就是一个计算图。
模型部署时通常把模型转换成静态的计算图，即没有控制流（分支语句、循环语句）的计算图。

ONNX 转换原理：
    从PyTorch 的模型到 ONNX 的模型，本质上是一种语言上的翻译。
    直觉上的想法是像编译器一样彻底解析原模型的代码，记录所有控制流。
    但前面也讲到，【我们通常只用 ONNX 记录不考虑控制流的静态图。】
    因此，PyTorch 提供了一种叫做追踪（trace）的模型转换方法：
    给定一组输入，再实际执行一遍模型，即把这组输入对应的计算图记录下来，保存为 ONNX 格式。
    【export 函数用的就是追踪导出方法】，需要给任意一组输入，让模型跑起来。
    我们的测试图片是三通道，256x256大小的，这里也构造一个同样形状的随机张量。
"""
print(f'开始生成中间表示——ONNX....')
x = torch.randn(1, 3, 256, 256)

onnx_path = r"./model/srcnn/srcnn.onnx"
with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        onnx_path,
        opset_version=11,  # ONNX 算子集的版本
        input_names=['input'],  # 是输入、输出 tensor 的名称
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

ort_session = onnxruntime.InferenceSession(onnx_path)
print(f'{type(input_img)}')
ort_inputs = {'input': input_img}  # 注意输入输出张量的名称需要和torch.onnx.export 中设置的输入输出名对应。
ort_output = ort_session.run(['output'], 
                             ort_inputs)[0]

# 后处理部分
ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort.png", ort_output)

print(f'DONE!')