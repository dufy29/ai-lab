""" 
TorchScipt 模型
与 torch.onnx 的关系？
与 torch.fx关系？

参考：
    TorchScript 解读（一）：初识 TorchScript, https://zhuanlan.zhihu.com/p/486914187

"""


import torch 
from torchvision.models import resnet18 
 
# 使用PyTorch model zoo中的resnet18作为例子 
model = resnet18() 
model.eval() 
 
# 通过trace的方法生成IR需要一个输入样例 
dummy_input = torch.rand(1, 3, 224, 224) 
 
# IR生成 
with torch.no_grad(): 
    """
    到这里就将 PyTorch 的模型转换成了 TorchScript 的 IR。这里我们使用了 trace 模式来生成 IR，
    所谓 trace 指的是进行一次模型推理，在推理的过程中记录所有经过的计算，将这些记录整合成计算图。
  
    """
    jit_model = torch.jit.trace(model, dummy_input) 
print(type(jit_model), '??????')

jit_layer1 = jit_model.layer1 
print(jit_layer1.graph) 
print(f'-'*20)
print(jit_layer1.code) 

print(f'-'*20)

# 调用inline pass，对graph做变换 
"""
上面代码中我们使用了一个名为inline的pass，将所有子模块进行内联，这样我们就能看见更完整的推理代码。
pass是一个来源于编译原理的概念，一个 TorchScript 的 pass 会接收一个图，
遍历图中所有元素进行某种变换，生成一个新的图。我们这里用到的inline起到的作用就是将模块调用展开，
尽管这样做并不能直接影响执行效率，但是它其实是很多其他pass的基础。
PyTorch 中定义了非常多的 pass 来解决各种优化任务，未来我们会做一些更详细的介绍
"""
torch._C._jit_pass_inline(jit_layer1.graph) 
print(jit_layer1.code) 

