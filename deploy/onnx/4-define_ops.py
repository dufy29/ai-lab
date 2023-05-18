""" PyTorch 中支持更多 ONNX 算子。

而要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证以下【三个环节】都不出错：
（不能出错的环节以及解决方法）
1，Pytorch算子：算子在 PyTorch 中有实现
    - 组合现有算子
    - 添加 TorchScript 算子 <--------------------
    - 添加普通 C++ 拓展算子  <--------------------
2，映射方法：有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
    - 为 ATen 算子添加符号函数  <---------------------
    - 为 TorchScript 算子添加符号函数
    - 封装成 torch.autograd.Function 并添加符号函数
3，ONNX算子：ONNX 有相应的算子
    - 使用现有 ONNX 算子
    - 定义新 ONNX 算子


参考：
    模型部署入门教程（四）：在 PyTorch 中支持更多 ONNX 算子
    https://zhuanlan.zhihu.com/p/513387413
"""


print(f'='*30)
print(f'映射问题及解决')
""" 
问题描述：
实际的部署过程中，我们都有可能会碰到一个最简单的算子缺失问题：
 算子在 ATen 中已经实现了，ONNX 中也有相关算子的定义，但是相关算子映射成 ONNX 的规则没有写。
在这种情况下，我们只需要为 ATen 算子补充描述映射规则的符号函数就行了
"""

import torch 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return torch.asinh(x) 
 
from torch.onnx.symbolic_registry import register_op 
 
def asinh_symbolic(g, input, *, out=None):
    # """
    # 符号函数，可以看成是 PyTorch 算子类的一个静态方法。在把 PyTorch 模型转换成 ONNX 模型时，
    # 各个 PyTorch 算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换。
    # 为算子【添加符号函数】一般要经过以下几步：
    #     获取原算子的前向推理接口。
    #     获取目标 ONNX 算子的定义。
    #     编写符号函数并绑定。

    # 输入（g, input, *, out=None）
    #     从除g（表示和计算图相关的内容）以外的第二个输入参数开始，其输入参数应该严格对应它在 ATen 中的定义：
    #     C:\Users\dufei\.conda\envs\pt1.8\Lib\site-packages\torch\_C\_VariableFunctions.pyi
    #     中定义如下：
    #     def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...

    # 输出：g.op("Asinh", input) 
    #     "Asinh"：算子在ONNX中的名称
    #     input ： 如在ONNX 算子文档里所见，这个算子只有一个输入，因此我们只要把符号函数的输入参数 input 对应过去就行。
    
    # """
    return g.op("Asinh", input) 

# 实现符号函数和原来的 ATen 算子“绑定”起来
# register_op的第一个参数是目标 ATen 算子名，第二个是要注册的符号函数，
# 第三个参数是算子的“域”，对于普通 ONNX 算子，直接填空字符串即可。
# 第四个参数表示向哪个算子集版本注册。我们遵照 ONNX 标准，向第 9 号算子集注册。
# 值得注意的是，这里向第 9 号算子集注册，不代表较新的算子集（第 10 号、第 11 号……）都得到了注册。
# 在示例中，我们先只向第 9 号算子集注册。
register_op('asinh', asinh_symbolic, '', 9)  

 
model = Model() 
input = torch.rand(1, 3, 10, 10) 
onnx_path = 'asinh.onnx'
torch.onnx.export(model, input, onnx_path)

print(f'验证自定义算子的正确性...')
import onnxruntime 
import torch 
import numpy as np 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return torch.asinh(x) 
 
model = Model() 
input = torch.rand(1, 3, 10, 10) 
torch_output = model(input).detach().numpy() 
 
sess = onnxruntime.InferenceSession(onnx_path) 
ort_output = sess.run(None, {'0': input.numpy()})[0] 
 
assert np.allclose(torch_output, ort_output)
print('验证通过')



# -----------------------------------------------
print(f'='*30)
print(f'Pytorch算子不支持，添加 TorchScript 算子，\
      以可变形卷积（Deformable Convolution）算子 为例')

import torch 
import torchvision 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = torch.nn.Conv2d(3, 18, 3) 
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3) 
 
    def forward(self, x): 
        return self.conv2(x, self.conv1(x)) 
 
from torch.onnx import register_custom_op_symbolic 
from torch.onnx.symbolic_helper import parse_args 
 
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none") 
def symbolic(g,  
        input, 
        weight, 
        offset, 
        mask, 
        bias, 
        stride_h, stride_w, 
        pad_h, pad_w, 
        dil_h, dil_w, 
        n_weight_grps, 
        n_offset_grps, 
        use_mask): 
    return g.op("custom::deform_conv2d", input, offset) 
 
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9) 
 
model = Model() 
input = torch.rand(1, 3, 10, 10) 
torch.onnx.export(model, input, 'dcn.onnx')


# -----------------------------------------------
print(f'='*30)
print(f'为 PyTorch 添加 C++ 拓展')
print(f'需要先执行 python .\setup.py develop   安装 my-add ')

import torch 
import my_lib 
class MyAddFunction(torch.autograd.Function): 
 
    @staticmethod 
    def forward(ctx, a, b): 
        return my_lib.my_add(a, b) 
 
    @staticmethod 
    def symbolic(g, a, b): 
        two = g.op("Constant", value_t=torch.tensor([2])) 
        a = g.op('Mul', a, two) 
        return g.op('Add', a, b) 
 
my_add = MyAddFunction.apply 
 
class MyAdd(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, a, b): 
        return my_add(a, b) 
 
model = MyAdd() 
input = torch.rand(1, 3, 10, 10) 
torch.onnx.export(model, (input, input), 'my_add.onnx') 
torch_output = model(input, input).detach().numpy() 
 
import onnxruntime 
import numpy as np 
sess = onnxruntime.InferenceSession('my_add.onnx') 
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0] 
 
assert np.allclose(torch_output, ort_output)

print('DONE!')

