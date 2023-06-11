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
    模型部署入门教程（五）：ONNX 模型的修改与调试
    https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/05_onnx_model_editing.md
"""



