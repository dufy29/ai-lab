
# 支持动态输入的超分辨率模型
## 1, onnx 导出时报错

~~~python
with torch.no_grad():
    torch.onnx.export(
        model,
        (x,3),
        onnx_path,
        opset_version=11,  # ONNX 算子集的版本
        input_names=['input', 'factor'],  # 是输入、输出 tensor 的名称
        output_names=['output'])
~~~
报错如下：
~~~
Traceback (most recent call last):
  File "c:/Users/dufei/codes/ai/ai-lab/deploy/onnx/1-srcnn-onnxRuntime copy.py", line 134, in <module>
    output_names=['output'])   <--------------------------------
  File "C:\Users\dufei\.conda\envs\pt10\lib\site-packages\torch\onnx\__init__.py", line 320, in export
..........................
  File "C:\Users\dufei\.conda\envs\pt10\lib\site-packages\torch\jit\_trace.py", line 132, in forward
    self._force_outplace,
  File "C:\Users\dufei\.conda\envs\pt10\lib\site-packages\torch\jit\_trace.py", line 118, in wrapper
  File "c:/Users/dufei/codes/ai/ai-lab/deploy/onnx/1-srcnn-onnxRuntime copy.py", line 53, in forward
    align_corners=False)   <--------------------------------
  File "C:\Users\dufei\.conda\envs\pt10\lib\site-packages\torch\nn\functional.py", line 3737, in interpolate
    return torch._C._nn.upsample_bicubic2d(input, output_size, align_corners, scale_factors)TypeError: upsample_bicubic2d() received an invalid combination of arguments - got (Tensor, NoneType, bool, list), but expected one of:
 * (Tensor input, tuple of ints output_size, bool align_corners, tuple of floats scale_factors)
      didn't match because some of the arguments have invalid types: (Tensor, NoneType, bool, list)
 * (Tensor input, tuple of ints output_size, bool align_corners, float scales_h, float scales_w, *, Tensor out)
~~~
原因：

刚刚的报错是因为 PyTorch 模型在导出到 ONNX 模型时，模型的输入参数的类型必须全部是 torch.Tensor。而实际上我们传入的第二个参数" 3 "是一个整形变量。这不符合 PyTorch 转 ONNX 的规定。我们必须要修改一下原来的模型的输入。为了保证输入的所有参数都是 torch.Tensor 类型的，我们做如下修改：

## 2,  TraceWarning 的警告

~~~
c:/Users/dufei/codes/ai/ai-lab/deploy/onnx/1-srcnn-onnxRuntime copy.py:53: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  scale_factor=upscale_factor.item(),
~~~
查看导出的onnx ，是有问题的(没有缩放因子输入)：

![](https://raw.githubusercontent.com/dufy29/ai-lab/main/pic/a.png)


解决方法：自定义算子

我们得自己定义一个实现插值的 PyTorch 算子，然后让它映射到一个我们期望的 ONNX Resize 算子上。

~~~python

class NewInterpolate(torch.autograd.Function): 
 
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
~~~

![](https://raw.githubusercontent.com/dufy29/ai-lab/main/pic/b.png)

经过自定义算子，导出的onnx 模型效果如下：
![](https://raw.githubusercontent.com/dufy29/ai-lab/main/pic/c.png)

参考：
- [第二章：解决模型部署中的难题](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/02_challenges.md)
- 