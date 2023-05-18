""" 
介绍 PyTorch 到 ONNX 的转换函数—— torch.onnx.export。

可以看成是一个算子翻译的过程，即从pytorch 算子翻译成ONNX 中定义的算子

ONNX 官方算子文档： https://github.com/onnx/onnx/blob/main/docs/Operators.md
掌握了如何查询 PyTorch 映射到 ONNX 的关系后，
我们在实际应用时就可以在 torch.onnx.export()的opset_version中先预设一个版本号，
碰到了问题就去对应的 PyTorch 符号表文件里去查。
如果某算子确实不存在，或者算子的映射关系不满足我们的要求，
我们就可能得用其他的算子绕过去，或者自定义算子了。


参考：
    模型部署入门教程（三）：PyTorch 转 ONNX 详解
    https://zhuanlan.zhihu.com/p/498425043
"""


print(f'='*30)
print(f'torch.onnx.export 调用讲解')
""" torch.onnx.export 调用讲解

torch.onnx.export中需要的模型实际上是一个torch.jit.ScriptModule。
而要把普通 PyTorch 模型转一个这样的 TorchScript 模型，
有跟踪（trace）和记录（script）两种导出计算图的方法。
torch.onnx.export默认使用【跟踪， trace】的方法导出。

"""

import torch 
 
class Model(torch.nn.Module): 
    def __init__(self, n): 
        super().__init__() 
        self.n = n 
        self.conv = torch.nn.Conv2d(3, 3, 3) 
 
    def forward(self, x): 
        for i in range(self.n): 
            x = self.conv(x) 
        return x 
 

models = [Model(2), Model(3)] 
model_names = ['model_2', 'model_3'] 
 
for model, model_name in zip(models, model_names): 
    dummy_input = torch.rand(1, 3, 10, 10) 
    dummy_output = model(dummy_input) 
    model_trace = torch.jit.trace(model, dummy_input) #  默认方法
    model_script = torch.jit.script(model) 
 
    # 跟踪法与直接 torch.onnx.export(model, ...)等价 
    torch.onnx.export(model_trace, dummy_input, f'tmp/{model_name}_trace.onnx', 
                      example_outputs=dummy_output) 
    # 记录法必须先调用 torch.jit.sciprt 
    torch.onnx.export(model_script, dummy_input, f'tmp/{model_name}_script.onnx',
                      example_outputs=dummy_output) 

# -----------------------------------------------
print(f'='*30)
print(f'设置动态维度......')
import torch 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv = torch.nn.Conv2d(3, 3, 3) 
 
    def forward(self, x): 
        x = self.conv(x) 
        return x 
 
 
model = Model() 
dummy_input = torch.rand(1, 3, 10, 10) 
model_names = ['tmp/model_static.onnx',  
'tmp/model_dynamic_0.onnx',  
'tmp/model_dynamic_23.onnx'] 
 
dynamic_axes_0 = { 
    'in' : [0], 
    'out' : [0] 
} 
dynamic_axes_23 = { 
    'in' : [2, 3], 
    'out' : [2, 3] 
} 
 
torch.onnx.export(model, dummy_input, model_names[0],  
input_names=['in'], output_names=['out']) 
torch.onnx.export(model, dummy_input, model_names[1],  
input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_0) 
torch.onnx.export(model, dummy_input, model_names[2],  
input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_23)

# -----------------------------------------------
print(f'='*30)
print(f'跟踪中断例子......')

"""
这些涉及张量与普通变量转换的逻辑都会导致最终的 ONNX 模型不太正确。
"""

class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        x = x * x[0].item() 
        return x, torch.Tensor([i for i in x]) 
 
model = Model()       
dummy_input = torch.rand(10) 
torch.onnx.export(model, dummy_input, 'tmp/a.onnx')


