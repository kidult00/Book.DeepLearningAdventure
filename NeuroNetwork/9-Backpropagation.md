# DeepLearning笔记：Backpropagation 反向传播算法

``阿扣``：今天我们来学习反向传播算法。

``阿特``：为什么你一脸严肃哦？

``阿扣``：咳咳，有吗……可能因为当初被 Backpropagation 这个词吓得不轻吧…… 反向传播算法是深度学习的核心之一，不过也没有很难，放轻松~

``阿特``：你是说你还是说我 😄

``阿扣``：来，我们先回忆一下，对多层神经网络，我们用梯度下降法去训练。之前已经学过如何计算输出节点的误差项 $\delta =(y-\hat y)f'(h)$，借助梯度下降算法，用误差项训练**隐层到输出层的权重**。

``阿特``：隐层到输出层。我记得最简单的神经网络应该有 3 层——是不是还有输入层到隐层？

``阿扣``：没错。

``阿特``：那该怎么求隐层节点对应的误差项呢？

``阿扣``：在神经网络里，输出节点的误差项，跟隐层的权重是成比例的。

``阿特``：意思是误差项越大，隐层节点的权重也越大？

``阿扣``：可以这么理解。既然我们知道输出的误差项，就可以用它来「反向传播」，求出隐层的误差项，再用于求输入节点的权重。

![](http://7xjpra.com1.z0.glb.clouddn.com/vlcsnap-2017-12-19-15h02m16s033.png)

``阿特``：咦，那不是反过来了？先知道输出结果，再反推输入权重？

``阿扣``：对的，所以叫做「反向」呀。

比如，输出层 k 个节点对应的误差项是 $\delta^o_k$ 。隐层有 j 个节点，那么隐层节点到输出节点的 j 个误差项是：

![](http://7xjpra.com1.z0.glb.clouddn.com/backprop-error.gif)

``阿特``：等等！先让我复习一下误差项是什么……

``阿扣``：嗯！误差项 δ 表示 ``误差 * 激活函数的导数``，$\delta_j=(y-\hat y)f'(h_j)$。对比一下 $\delta^h_j=\sum W_{jk} \delta^o_k f'(h_j)$，看看有什么不同？

``阿特``：隐层到输出层的误差 (y-y^) 变成了 $\sum W_{jk} \delta^o_k$

``阿扣``：很棒！你发现了吧，$\delta_k$ 成为了 wx + b 中的变量 x：

![](http://7xjpra.com1.z0.glb.clouddn.com/vlcsnap-2017-12-19-15h06m32s756.png)

``阿特``：啊，又要来算这个了……

``阿扣``：没关系，虽然看上去麻烦一些，但是跟正向传播的做法很类似，权重的更新为 $\Delta w_{ij}=\eta \delta^h_jx_i$ 。

``阿特``：每次都要来一遍，要死不少脑细胞啊……

``阿扣``：那我给你列个清单吧，每次照着做就好。

假设我们考虑最简单的神经网络：只有一个隐层节点，只有一个输出节点。用反向传播算法更新权重的算法如下：

- 给每一层的权重赋值为 0
  + 输入层→隐层的权重 $\Delta w_{ij}=0$
  + 隐层→输出层的权重 $\Delta W_j=0$
​
- 对训练集里的每一个数据：
  + 使用 forward pass，计算输出节点的值 $\hat y$
  + 计算输出节点的误差梯度 $\delta^o=(y-\hat y)f'(z)$，  这里的 $z=\sum_jW_ja_j$
  + 将误差反向传递到隐层 $\delta^h_j=\delta^oW_jf'(h_j)$
  + 更新权重步长
    * $\Delta W_j = \Delta W_j + \delta^oa_j$
    * $\Delta w_{ij} = \Delta w_{ij} + \delta^h_ja_i$
- 更新权重（η 为学习率，m 为输入节点的个数):
  + $W_j = W_j + \eta \Delta W_j /m$
  + $w_{ij} = w_{ij} + \eta \Delta w_{ij} /m$
- 重复 e 次训练步骤 (epochs)

``阿特``：天！看上去好复杂。

``阿扣``：练习两次就能熟悉起来了，别担心。下一次我带你用 Python 实现反向传播算法。


### Ref

- [Deep Learning Nanodegree | Udacity](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)
- [Yes you should understand backprop – Medium](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
- [CS231n Winter 2016 Lecture 4 Backpropagation, Neural Networks 1-Q_UWHTY_TEQ.mp4 - YouTube](https://www.youtube.com/watch?v=59Hbtz7XgjM)
