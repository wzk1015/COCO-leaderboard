# COCO Leaderboard

Notes of top entries of COCO learderboard based on [paperswithcode](https://paperswithcode.com/sota/object-detection-on-coco) (up to 2022.3.22, deduplicated)

References: several zhihu articles (see my [collection list](https://www.zhihu.com/collection/792178490)) and CSDN blogs



[TOC]



## Summary

### Leaderboard

*Italic* denotes unique contribution

| Rank | Method            | AP box | Extra Data (Detection)                                     | Pretrain Data                    | Backbone            | Detector                  | Training                              | Scale  |
| ---- | ----------------- | ------ | ---------------------------------------------------------- | -------------------------------- | ------------------- | ------------------------- | ------------------------------------- | ------ |
| 1    | DINO              | 63.3   | Object365                                                  | IN22K                            | SwinL               | *DINO* (DETR)             |                                       | multi  |
| 2    | SwinV2            | 63.1   | Object365                                                  | IN22K-ext-70M                    | *SwinG*             | HTC++                     |                                       | multi  |
| 3    | Florence          | 62.4   | FLOD9M (COCO, O365, LVIS, OpenImages), IN22K self-training | FLD900M                          | SwinH               | DyHead                    | *multimodal, multi-task*              | multi  |
| 4    | GLIP              | 61.5   | FourODs, GoldG+                                            | 27M grounding data               | SwinL               | DyHead                    | *object-word contrastive pretraining* | multi  |
| 5    | Soft Teacher      | 61.3   | Object365                                                  | IN22K                            | SwinL               | HTC++                     | *semi-supervised*                     | multi  |
| 6    | DyHead            | 60.6   | IN22K self-training                                        | IN22K                            | SwinL               | *DyHead* (Mask RCNN)      |                                       | multi  |
| 7    | CBNetV2           | 60.1   |                                                            | IN22K                            | SwinL               | HTC                       | *composite backbone*                  | multi  |
| 8    | Focal Transformer | 58.9   |                                                            | Swin init, finetune on IN1K      | *Focal Transformer* | DyHead                    |                                       | multi  |
| 9    | YOLOR             | 57.3   |                                                            | IN1K                             | CSPDarkNet-53       | *YOLOR* (YOLOv4-CSP)      |                                       | single |
| 10   | CopyPaste         | 57.3   | self-training (unlabeled COCO, O365)                       | IN (self-training checkpoint)    | EfficientNet        | FPN                       | *data augmentation*                   | single |
| 11   | SOLQ              | 56.5   |                                                            | IN22K                            | SwinL               | *SOLQ* (DETR)             |                                       | single |
| 12   | CenterNet2        | 56.4   |                                                            | IN1K                             | Res2Net101-DCN      | *CenterNet2 (CenterNet)*  | *anchor free*                         | single |
| 13   | QueryInst         | 56.1   |                                                            | IN22K                            | SwinL               | *QueryInst* (DETR)        |                                       | single |
| 14   | Scaled YOLOv4     | 55.8   |                                                            | IN1K                             | CSPDarkNet-53       | *scaled YOLOv4* (YOLOv4)  | *network scaling approach*            | single |
| 15   | DetectoRS         | 55.7   |                                                            | IN                               | ResNeXt101          | *DetectoRS* (HTC)         |                                       | multi  |
| 16   | Mish              | 55.2   |                                                            | IN1K                             | CSPDarkNet-53       | YOLOv4                    | *activation function*                 | multi  |
| 17   | Self-training     | 54.3   |                                                            | IN1K, OpenImages (self-training) | SpineNet-190        | RetinaNet                 | *self-training*                       | single |
| 18   | UniverseNet       | 54.1   |                                                            | IN1K                             | Res2Net101-DCN      | *UniverseNet* (RetinaNet) |                                       | multi  |
| 19   | EfficientDet      | 53.7   |                                                            | IN1K                             | EfficientNet        | *EfficientDet* (FPN)      |                                       | single |
| 20   | PAA               | 53.5   |                                                            | IN                               | ResNeXt-152-DCN     | *PAA* (RetinaNet)         | *anchor assignment*                   | multi  |
| 21   | LSNet             | 53.5   |                                                            | IN1K                             | Res2Net-101-DCN     | *LSNet* (FPN)             |                                       | multi  |
| 22   | ResNeSt           | 53.3   |                                                            | IN1K                             | *ResNeSt-200*       | Cascade-RCNN              |                                       | multi  |
| 23   | GFLv2             | 53.3   |                                                            |                                  | Res2Net-101-DCN     | *Generalized Focal Loss*  | *loss function*                       | multi  |
| 24   | RelationNet++     | 52.7   |                                                            |                                  | ResNeXt-101-DCN     | *RelationNet++*           |                                       | multi  |
| 25   | Deformable DETR   | 52.3   |                                                            | IN1K                             | ResNeXt-101-DCN     | *Deformable DETR* (DETR)  |                                       | multi  |

### Related Papers (unordered)

* TODO

1. SEPC
2. Rethinking ImageNet Pre-Training

* Done

1. ResNet
2. RCNN
3. Fast RCNN
4. Faster RCNN
5. Mask RCNN
6. FPN
7. Focal Loss (RetinaNet)
8. YOLOv4
9. DETR
10. Deformable Conv
11. SimMIM
12. DN-DETR
13. DAB-DETR
14. HTC
15. STAC
16. FixMatch
17. CenterNet
18. Swin Transformer
19. CLIP
20. EfficientNet
20. Res2Net
20. ResNeXt
20. CSPNet
20. NAS-FPN
20. PANet
20. ATSS
27. FCOS
28. RepPoints
29. Sparse RCNN



## Methods

### 1. DINO

<img src="coco%20leaderboard.assets/image-20220323160918699.png" alt="image-20220323160918699" style="zoom:50%;" />

基于DETR提出改进：

1. Contrastive DeNoising Training

在DN-DETR的基础上增加了对比学习，即正负样本同时添加噪声，对于每个anchor query需要完成降噪、判断有无物体。添加smaller noise的作为正样本，其他作为负样本，主要目的是改善box匹配，加速训练

2. Mixed Query Selection

与之前工作把anchor和query都设成静态（DETR；query太多，计算量大且难以收敛）或都设成动态（Deformable DETR；可能有多个物体或只有物体的一部分）不同，把Content Queries设为静态（可学习），Anchor Boxes设为动态（由encoder初始化）

3. Look Forward Twice

box refine时增加前一层的特征作为输入，类似于增大感受野



### 2. SwinV2

<img src="coco%20leaderboard.assets/image-20220323163649806.png" alt="image-20220323163649806" style="zoom:50%;" />

Swin V1的问题：

1. 随着层数加深，深层激活值提升，特征方差变大，变得不稳定
2. 分辨率改变时性能下降



改进：

1. 对于方差大的问题，将LayerNorm移动到MLP/Attention之后，残差连接之前
2. 对于attention map易被少数像素主导的问题，将attention计算方式改成scaled cosine
3. 对于跨分辨率问题，提出log continuous position bias，减少插值比率

4. 一些工程技巧减少显存占用
5. 额外数据：ImageNet-22K扩展五倍，自监督训练（SimMIM）



### 3. Florence

<img src="coco%20leaderboard.assets/image-20220323173214714.png" alt="image-20220323173214714" style="zoom: 33%;" />

计算机视觉任务:分类、检索、检测、VQA、Image Caption、视频检索、动作识别

迁移学习：finetune，linear probing，few shot，zero shot

与CLIP等多模态模型相比，扩展到了object级别、视频特征



图片encoder使用swin等，文本使用roberta，进行Unified Constrastive Learning。检测部分使用Dynamic Head模型



### 4. GLIP

<img src="coco%20leaderboard.assets/image-20220323180325391.png" alt="image-20220323180325391" style="zoom:50%;" />

统一phrase grounding和detecion，利用grounding数据进行物体和文本的对比学习

与CLIP不同，使用Deep Fusion融合文本和视觉encoder

可以zero shot预测训练数据中未出现的类别，在LVIS上也有提升



### 5. Soft Teacher

<img src="coco%20leaderboard.assets/image-20220324130948314.png" alt="image-20220324130948314" style="zoom:50%;" />

半监督目标检测，使用伪标签，相比于STAC特色是端到端

在每个训练 step 中同时进行生成伪标签、利用伪标签对无标签图片进行训练。在每个 batch中随机抽样有标签和无标签数据，student模型进行检测训练，teacher模型（student的EMA平均）对无标签数据生成伪标签

类似FixMatch，无监督数据进行强弱两种增强，分别进入student和teacher。性能依赖于伪标签的质量

cls伪标签：采用soft teacher，使用teacher模型的分类分数来加权student模型预测proposal的loss。低阈值+soft teacher的效果好于高阈值

reg伪标签：采用box jittering，多次抖动伪前景框proposal，利用teacher模型进行回归，将回归框的方差用作可靠性度量，高可靠性（方差低）的proposal用于student定位分支的训练



### 6. Dynamic Head

<img src="readme.assets/image-20220324142014840.png" alt="image-20220324142014840" style="zoom:50%;" />

在一个head里同时实现三种感知。对feature pyramid连续进行三种attention操作，分别针对scale, spatial和task，即layer, space(H,W), channel三个维度

dynamic head模块可以添加到one-stage和two-stage检测器中，one-stage中放在FPN backbone后、class box center的分类器前，two-stage中将最后一个task-aware attention放在ROI Pooling后、class box分类器前，其余的放在ROI Pooling前



### 7. CBNetV2

<img src="readme.assets/image-20220324150214078.png" alt="image-20220324150214078" style="zoom:50%;" />

将多个相同结构Backbone使用Dense Higher-Level Composition的方式组合在一起，每个layer给所有更浅layer提供输入

为了避免深度带来的训练困难，在训练时使用assisting backbone（前K-1个backbone中的每个）的输出训练同样权重的detection head2，loss作为assistant supervision



### 8. Focal Transformer

![image-20220325212755763](readme.assets/image-20220325212755763.png)

粗粒度导致sub-optimal，细粒度导致计算开销过大。以细粒度的方式关注离自己近的token，以粗粒度的方式关注离自己远的token。将较远的patch进行pooling聚合，增大感受野

但显存占用和计算量较大，对高分辨率的预测任务不友好



### 9. YOLOR

![image-20220325222448340](readme.assets/image-20220325222448340.png)

提出了一个统一的网络来同时编码显式知识和隐式知识，在网络中执行了核空间对齐、预测细化和多任务学习来学习隐式知识，同时对多个任务形成统一的表示

隐式知识：

1. 利用投影向量的内积和隐式表示来达到降低流形空间维度的目的
2. 对输出特征和隐式表征进行加法和乘法运算，这样就可以对核空间进行变换、旋转和缩放，以对齐神经网络的每个输出核空间
3. 引入加法，预测中心坐标的偏移；引入乘法，搜索锚框的超参数集；分别使用点乘和concat来执行多任务特征选择

传统神经网络的表示空间专注于特定任务，对于其他任务不变。通用的网络希望对于所有潜在任务都有效。

特征对齐：在FPN的每一个特征映射层添加隐式知识进行特征对齐

目标检测预测细化：在YOLO的每一个输出层添加隐式知识进行预测细化

为训练多任务训练一个规范的表征，通过给每个任务分支引入隐式表征增强表征能力

总结：引入隐式知识，仅增加不到万分一的参数量和计算量，模型性能得到有意义的提升，同时收敛更快。



### 10. CopyPaste

<img src="readme.assets/image-20220328005557387.png" alt="image-20220328005557387" style="zoom:50%;" />

将一个图片的物体直接复制到另一个图片。与之前的复制粘贴方法不同，不需要对上下文建模获取粘贴位置

未使用旋转、高斯模糊，使用了水平翻转、大尺度抖动

作为半监督训练方法：对未标记数据生成伪标签、将ground truth实例粘贴进来作为新数据

baseline模型在LVIS 2020挑战赛获胜者的基础上提升3.6%



### 11. SOLQ

<img src="readme.assets/image-20220328010758298.png" alt="image-20220328010758298" style="zoom:50%;" />

在DETR上增加预测mask的head。其中主要贡献是UQR模块，将mask从二维feature map压缩编码到一维向量，从而可以用head预测。测试时将一维向量解码为mask。编解码方式有Sparse Coding, PCA, DCT



### 12. CenterNet2

<img src="readme.assets/image-20220328162837596.png" alt="image-20220328162837596" style="zoom:50%;" />

one-stage检测器：依赖于单独分类和回归分支；类别很多时（如LVIS）速度不再比two-stage快

two-stage 检测器：相对较弱的RPN，召回排名前1K的proposal，减慢了速度

CenterNet2框架（概率两阶段模型）将一个强大的One-Stage Detector与后面的分类阶段集成在一起，在第1阶段使用更少、质量更高的预选框，以输出更高的准确性和更快的速度

第1阶段预测class agnostic概率，第2阶段预测分类分数，进行相乘

使用FPN将CenterNet升级到多尺度。然后一阶段使用CenterNet，二阶段使用Faster RCNN / Cascade RCNN



### 13. QueryInst

![image-20220328171331940](readme.assets/image-20220328171331940.png)

构建有效的基于 Query 的端到端实例分割算法

将DETR中的transformer block加入mask head，box和mask进行多阶段迭代优化，并将query作为输入

对于Cascade Mask R-CNN和HTC，在统计意义上细化了不同阶段proposal的质量。对于每个阶段，训练样本的数量和分布都有很大的不同，在不同阶段中，每个单独的proposal没有明确的内在对应

对于QueryInst，各个阶段之间的连接是通过query中固有的一一对应自然建立的这种方法消除了显式多阶段mask head连接和proposal分布不一致的问题



### 14. Scaled YOLOv4

<img src="readme.assets/image-20220328202340702.png" alt="image-20220328202340702" style="zoom:50%;" />

提出了一种网络缩放方法，修改深度，宽度，分辨率，还可以修改网络的结构

对yolov4进行了重新设计，提出了YOLOv4-CSP，然后基于onYOLOv4-CSP开发了scale - yolov4。新的架构可以有效地减少ResNet、ResNeXt和Darknet上的FLOPs，分别减少23.5%、46.7%和50.0%

对于tiny模型，控制计算量、feature map尺寸、保持卷积后通道数相同、最小化卷积输入/输出

对于large模型，更好地预测物体大小的能力基本上取决于特征向量的感受野



### 15. DetectoRS

<img src="readme.assets/image-20220328204437930.png" alt="image-20220328204437930" style="zoom:50%;" />

在宏观层面上，提出了Recursive Feature Pyramid，将FPN的额外反馈连接加入到自下而上的backbone层中。使用带有空洞卷积的空间金字塔池化（ASPP）模块来实现两个递归特征金字塔的级联连接。融合模块通过卷积层和Sigmoid运算来计算注意力图，然后将结果注意力图用于计算更新后的加权和

在微观层面上提出Switchable Atrous Convolution，以不同的空洞率进行卷积，使用switch函数合并。将主干网络ResNet中的每个3x3卷积层都转换为SAC，从而在不同的空洞率之间实现对卷积计算的软切换



### 16. Mish

<img src="readme.assets/image-20220328220206062.png" alt="image-20220328220206062" style="zoom:50%;" />

提出了新激活函数：$Mish(x)=x · tanh(ln(1+e^x))$

无边界(即正值可以达到任何高度)避免了由于封顶而导致的饱和。对负值的轻微允许允许更好的梯度流，而不是像ReLU中那样的硬零边界。

平滑的激活函数允许更好的信息深入神经网络，从而得到更好的准确性和泛化

尽管如此，测试了许多激活函数，大多数都无法执行。主要区别可能是Mish函数在曲线上几乎所有点上的平滑度



### 17. Rethinking Pre-training and Self-training

<img src="readme.assets/image-20220329125408579.png" alt="image-20220329125408579" style="zoom:50%;" />

延续何恺明的Rethinking ImageNet Pre-Training，通过大量的实验验证了在检测和分割任务中 Pre-training 和 Self-training 的效果

- 对于 Pre-training

- - 当使用更强大的数据增强方案时，预训练反而会损害模型的性能
  - 可用标注数据越多，预训练的作用越小，价值越低

- 对于 Self-training

- - Self-training 在使用高强度数据增强策略时仍能够提升性能，即使当预训练已经会损害性能时
  - Self-training 在各种标注数据规模下都具能提升性能，并可以与预训练共同起作用

- 对于 Self-supervised pre-training 同样在高强度数据增强策略时会损害性能

数据增强：使用四种不同的数据增强策略，依次增强，第一种是和Rethinking ImageNet Pre-Training中一致，后三种来自 AutoAugment 和 RandAugment

self-training：能提升性能，但相对于预训练来说会花1.3x-8x的时间。首先在有监督数据训练teacher模型，然后在无监督数据上生成伪标签，最后训练student基于监督标签和伪标签优化。并设计了一种loss正则化的方法

作者猜测预训练效果不佳是由于 pre-training 无法感知检测或分割任务感兴趣的地方并且无法适应，例如 ImageNet 上训练好的特征可能忽视了检测任务所需的位置信息

联合训练：表示同时训练 ImageNet 分类和 COCO 的对象检测，实验结果 表明使用 350 epochs 的预训练模型进行初始化能够取得 +2.6AP的提升，但使用联合训练只需要在 ImageNet 上训练 19 epochs 就能获得 +2.9AP 的提升。此外预训练，自训练，联合训练加一起能够得到更大的提升。

对齐任务：self-training 能够对齐任务，之前的实验中 Open Images 不能提升 COCO 性能，但附录的实验中self-training 能够用 Open Images 提升 COCO 指标



### 18. USB

<img src="readme.assets/image-20220329133159754.png" alt="image-20220329133159754" style="zoom:50%;" />

提出了Universal-Scale object detection benchmark，将COCO, Waymy Open Dataset, Manga109组合起来，包含不同尺寸的物体。以及用于训练和评估的protocols、指标等

UniverseNet: 在RetinaNet基础上，使用ATSS, SEPC, DCN，多尺度训练，加入GFL, SyncBN, iBN



### 19. EfficientDet

<img src="readme.assets/image-20220329135309572.png" alt="image-20220329135309572" style="zoom:50%;" />

BiFPN：结合了NAS-FPN和PANet，保留了双向融合和跳级连接，变为可重复block。采用加权（权重可学习）平均的方法融合不同层的特征

模型scaling方法：涉及到backbone的网络深度、宽度、输入尺寸，BiFPN的深度、宽度，box/class预测网络的深度、宽度。因为搜索空间大，人为地缩小了搜索空间，然后使用启发式的方法来确定upscale的参数



### 20. PAA

<img src="readme.assets/image-20220329145325705.png" alt="image-20220329145325705" style="zoom:50%;" />

此前方法中anchor标签分布主要根据IOU阈值，会受限于bbox内的背景噪声，还会造成训练和测试时的不一致

本文提出了一个给anchor分配标签的概率模型，和一个后处理的方法，优势如下：

1. 分配标准取决于分类精度和IOU的组合，而不单单取决于IOU，这对model的增益更大（focus分类精度使得这个metric可以考虑box本身的内容）
2. 通过混合高斯分布GMM对正负样本进行建模，可解释性强。正样本和负样本的分数分布分别用一个高斯分布刻画，通过EM算法优化
3. 丢掉了IOU阈值，正样本数目等超参数，模型更稳定
4. 通过修改网络结构对box与GT的IOU进行预测（在class和box预测头上增加一个head），训练阶段有真实的IOU作为label，测试阶段可以预测IOU，使得训练和测试阶段保持一致



### 21. LSNet

<img src="readme.assets/image-20220329152618672.png" alt="image-20220329152618672" style="zoom:50%;" />

将目标检测、实例分割、姿态估计三个任务统一。首先预测中心点，然后对三个任务分别预测四个极值点向量/N个轮廓点向量、13个关键点向量。设计了Cross IoU loss将IoU降低到一维进行计算，效果好于smooth L1 loss

回归过程采取了init+refine的方式。先回归一组初始的向量，这些向量已经比较接近关键点。利用Pyramid DCN获取这些关键点处的特征，也就是说DCN不仅只在目标所在的FPN层计算，还会把DCN的offsets等比例映射至相邻的FPN层进行计算，将三层的所得特征相加，形成最终的关键点特征。利用这些特征再预测一组向量，两组向量叠加最终形成预测向量



### 22. ResNest

<img src="readme.assets/image-20220329155413607.png" alt="image-20220329155413607" style="zoom:50%;" />

在SENet，SKNet，ResNeXt的基础上提出了Split-Attention模块，用于替换ResNet中四个stage的conv。在分类、检测、分割上都有非常明显的提升。Split-Attention即在每个分组内用attention
