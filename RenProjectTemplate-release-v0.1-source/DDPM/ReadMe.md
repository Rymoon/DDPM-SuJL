

# ReadMe


**当前版本说明**


有空再说吧（咕咕咕）


## 环境


### 安装Python依赖

`conda install -f conda-env.yml`


`.py`脚本在windows和linux均可运行，因为python是跨平台的。`.bat`脚本用于windows，`.sh`用于linux。



### 安装CUDA Toolkit + cudnn

CUDA版本首先适配GPU型号（比如，RTX3060至少11.1），然后适配Pytorch版本。前者必须满足，后者可以凑合。

下载安装CUDA Toolkit

注册nvidia，下载对应CUDA版本的cudnn，解压放在指定位置。


### 安装本程序包 RenNet

cd 到 含有setup.py的目录。执行

````bash
pip install -e .
````


同时依赖RenNet包，同样cd到MyHumbleADMM7有setup.py的目录，执行

````bash
pip install -e .
````


## 快速上手

默认包含`RenPINN`文件夹的目录为根目录`/`。

### 1. 配置路径

创建或修改`/RenPINN/env.json`，设置训练环境，包括文件路径、cpu核心数等等。

详细参见env.md文件

````json
{
"result_folder_name":"Results",
"temp_folder_name":"Temp"
}
````

### 2.文件结构

#### 环境设置/RenPINN/:

- env.json 配置文件。文件路径 、 cpu核心数等。
- env.md 说明
- env.py 实际处理

#### 程序框架 /RenPINN/core:

- **实际调用 RenNet (...) 包的核心库。**

#### 程序主体 /RenPINN/apps/wave1d：

#### 训练结果/Results/

- GoodResults.md： Markdown文件（文本）。好结果的实验
- 实验文件夹的名字与训练脚本一致

#### 批量训练脚本/Scripts/

- bash脚本用于Ubuntu
- bat脚本用于Windows




## 常见问题

### Q: 执行有GUI的程序，报以下错：

````bash
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, minimal, minimalegl, offscreen, vnc, webgl, xcb.
````

GUI用的是PySide2. 如果在vscode的built-in terminal 中出这个错，那就试试mobaxterm的终端。 前者的X11server有问题。


### Q: loss 为Nan

*A*: 一定是网络中，某一步的梯度是inf或nan了。可能有以下原因。

	1. 注意，对最后一层不加激活函数的网络来说，其可能输出任何值。例如，过程中有`y=sqrt(x)`，则梯度在`0+`附近趋于`+inf`。 Pytorch不会对此类异常报警。
	1. 对于LBFGS来说，有时需要float64，而不是float32.

### Q: GPU显存不足

*A*: 减小网络深度，减小样本切片，减小batch_size

### Q: 结果好坏的标准

*A*: 对于图像分割，我们使用AUC(Area under ROC curve)，是二分类任务的指标。而对于这个分割任务，AUC直观上可以看作预测和真值的相似度，0是不同，1是相同，越大越好。训练时显示的是test set上的平均值，用sklearn提供的函数。你也可以输出图片，然后自己在jupyter notebook 里测AUC。

### Q:如何暂停，有没有断点？

*A*: 对于 pytorch_lightning, 按ctrl+C会强制中止当前`trainer`，并继续执行。比如，

```python
trainer1.fit(model) # 1. <--- ctrl+c
trainer2.fit(model) # 2.
trainer2.test(model)# 3.
```

则会继续执行 `#2.`, `#3.`。


### Q:拷贝后如何设置路径？是绝对路径吗？

*A*: 在`RenNet/env.json`中设置，参见`env.md`中的介绍。这里的路径将用于加载数据集、保存训练数据等。

### Q: 如何设置程序运行的GPU设备id？如何获得命令行帮助

*A*: 在`pl.trainer(...)`提供合适的参数。详见`pytorch_lightning`的文档。

## 附录

### Q: torch.rfft在pytorch 1.7之后被废弃，如何替代？

*A*: 旧版pytorch中torch.rfft和irfft在新版本中的对应

总结
pytorch旧版本（1.7之前）中有一个函数torch.rfft()，但是新版本（1.8、1.9）中被移除了，添加了torch.fft.rfft()，但它并不是旧版的替代品。

傅里叶的相关知识都快忘光了，网上几乎没有相关资料，看了老半天官方文档，终于找到了对应的函数。

虽然整个过程的细节我还没有完全搞懂，但是网上相关的资料实在太少了，所以我把我都知道的都写上来，希望对后来者有点帮助

结论直接跳到最后

rfft
旧版pytorch
首先，我是在一篇论文的开源代码里面看到的旧版rfft

label_fft1 = torch.rfft(label_img4, signal_ndim=2, normalized=False, onesided=False)

label_img4为一张图片，dtype=torch.float32，size比如为[1,3,64,64]

旧版本中torch.rfft()的参数说明为

input (Tensor) – the input tensor of at least signal_ndim dimensions
signal_ndim (int) – the number of dimensions in each signal. signal_ndim can only be 1, 2 or 3
normalized (bool, optional) – controls whether to return normalized results. Default: False
onesided (bool, optional) – controls whether to return half of results to avoid redundancy. Default: True
在上述的代码中，signal_ndim=2 因为图像是二维的，normalized=False 说明不进行归一化，onesided=False 则是希望不要减少最后一个维度的大小

在1.7版本torch.rfft中，有一个warning，表示在新版中，要“one-side ouput”的话用torch.fft.rfft()，要“two-side ouput”的话用torch.fft.fft()。这里的one/two side，跟旧版的onesided参数对应，所以我们要的是新版的torch.fft.fft()


需要注意的是，假设输入tensor的维度为  ，则输出tensor的维度为 （多一个维度） 。最后一个维度2表示复数中的实部、虚部，即  这样的复数，在旧版pytorch中表示为一个二维向量 

新版pytorch
新版pytorch中，各种在新版本中各种fft的解释如下

````
fft, which computes a complex FFT over a single dimension, and ifft, its inverse the more general fftn and ifftn, which support multiple dimensions The “real” FFT functions, rfft, irfft, rfftn, irfftn, designed to work with signals that are real-valued in their time domains
The “Hermitian” FFT functions, hfft and ihfft, designed to work with signals that are real-valued in their frequency domains
Helper functions, like fftfreq, rfftfreq, fftshift, ifftshift, that make it easier to manipulate signals
````

可以看到这里也有rfft，官方文档说是用来处理都是实数的输入。但是它在前面的warning中说了是one-side，而我们要的是two-side。此外实数也可以看作是虚部都为0的复数，所以用fft没问题

新版的rfft和fft都是用于一维输入，而我们的图像是二维，所以应该用rfft2和fft2。在fft2中，参数dim用来指定用于傅里叶变换的维度，默认(-2,-1)，正好对应H、W两个维度。

新版所有的fft都不将复数  存成二维向量了，而是一个数 ，数据类型为complex，输出tensor的维度还是  。所以如果要跟旧版中一样存成二维向量（多一个维度），需要用.real()和.imag()提取复数的实部和虚部，然后用torch.stack()堆到一起（具体操作方式见总结）

irfft

同理新版pytorch中，torch.fft.ifft2()对应旧版中torch.irfft(xxx, signal_ndim=2, onesided=False)；torch.fft.irfft2()对应旧版中torch.irfft(xxx, signal_ndim=2, onesided=True)

同理，如果旧版中signal_ndim不管为多少，新版中都可以用torch.fft.ifftn()和torch.fft.irfftn()

需要注意的是， 新版中要求输入的数据类型为complex，即要求输入的维度不跟旧版一样将复数的实部和虚部存成二维向量（即在最后多出一个值为2的维度）。如果说输入时以二维向量存复数，则需要使用torch.complex()将其转化成complex类型（具体操作方式见总结）



总结

```python
import torch
input = torch.randn(1, 3, 64, 64)
# 旧版pytorch。参数normalized对这篇文章的结论没有影响，加上只是跟开头同步
output_fft_old = torch.rfft(input, signal_ndim=2, normalized=False, onesided=False)
output_ifft_old = torch.irfft(output_fft_old , signal_ndim=2, normalized=False, onesided=False)
# 新版
output_fft_new = torch.fft.fft2(input, dim=(-2, -1))
output_fft_new_2dim = torch.stack((output_fft_new.real, output_fft_new.imag), -1)
output_ifft_new = torch.fft.ifft2(torch.complex(output_fft_new_2dim[..., 0], output_fft_new_2dim[..., 1]), dim=(-2, -1))    # 如果运行了torch.stack()
output_ifft_new = torch.fft.ifft2(output_fft_new_2dim, dim=(-2, -1))    # 没有运行torch.stack()
```

编辑于 2022-03-23 13:54
