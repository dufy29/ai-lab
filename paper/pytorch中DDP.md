
DDP 设计模块：
![](https://raw.githubusercontent.com/dufy29/ai-lab/main/pic/a11.png)  

DNN完整的迭代训练包括3个步骤：  
前向计算loss（pytorch构建 autograd图记录前向执行）  
---> 反向计算梯度（使用autograd 图进行反向传播）   
... 梯度同步（AllReduce）     
---> optimizer step更新参数（optimizer优化器使用梯度更新参数）

~~~
使用32-GPU节点，共256GPU 做了试验，覆盖NCCL, Gloo，结论：   
1）通信是主要的训练开销，尤其是模型尺寸越大时候
2）bucket size 会显著影响通信效率，要设置的合理
~~~

pytorch 提供的加速DDP策略： 
- bucketing gradients, 
- overlapping computation with communication,
- and skipping gradient synchronization.

pytorch提供的分布式方案：   
- DP, 单进程多线程，单节点    
- DDP，多进程，可以跨节点     
- RPC（RPC-Based Distributed Training ），模型并行
e.g., pipeline并行，parameter server

~~~
DDP采用`AllReduce` 通信原语计算梯度和，NCCL/Gloo/MPI等通信库均支持。
比较高效的算法：ring-based `AllReduce` 、 tree-based `AllReduce`
`AllReduce` 属于同步通信，参数服务中的 P2P（点对点通信）属于异步通信
~~~

~~~
sd
~~~