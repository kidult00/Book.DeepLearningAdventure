从前，有一对失散多年的双胞胎，叫做阿特和阿扣，让他俩来带我们穿越深度学习的重重迷雾吧！

阿特——十万个为什么懵懂星人，富有创造力的好奇宝宝
阿扣——外萌内冷，不说人话的学霸，经常跟机器谈心

## Neural Networks 神经网络
``阿特``：听说深度学习的思想受到神经网络的启发，那是什么玩意儿？

``阿扣``：神经网络有生物神经网络和人工神经网络两种。在生物神经网络中，每个神经元与其他神经元相连，当它接收外界信号（其他神经元）作为输入，当电位超过了某个阈值（threshold）而被「激活」时，会向相连的神经元「发射」（fire）信号。

![](http://7xjpra.com1.z0.glb.clouddn.com/neuralNetwork.png)

``阿特``：那跟机器有关系吗？机器没有生命啊……

### Perceptrons 感知器

``阿扣``：在计算机科学中，我们将独立的计算单元看做神经元。感知机 (Perceptron) 是神经网络的基本单位。每一个感知机都完成一个类似「给我一个数字，我告诉你它是正还是负」这样的简单任务。

比如说，我们把神经元看做包含一个 0 到 1 之间数字的小球：

0.8

神经元里面的数字我们叫它 激活函数 (Activation)。当数字超过某个阈值，比如说 0.5 时，我们就说这个神经元被激活了，它会输出 1 作为信号。如果神经元包含的数字小于 0.5，那它就输出 0，表示没有被激活。

这个神经元就是一个感知机。一个感知机接收若干二进制输入 $x_1,x_2,...$，然后产生一个二进制输出：

![](http://neuralnetworksanddeeplearning.com/images/tikz0.png)

``阿特``：这小球长得倒是有那么一丢丢像神经元……

``阿扣``：这个最简单的系统里，包含：

- 输入：这个神经元接收到的其他神经元的信号
- 判断器：激活函数
- 输出：1 表示 yes「发射」，0 表示 no「不发射」

``阿特``：艾玛，这也叫简单？

``阿扣``：它其实是这个意思：

![](https://viniciusarruda.github.io/images/mp_neuron.png)

``阿特``：好吧我错了……让我晕一晕

``阿扣``：其实主要看蓝色的字就好。神经元怎么计算输出呢？我们引入「权重」(weights)，它表示从输入到输出的重要程度。权重的和 $\sum_j w_jx_j$ 如果大于阈值 $v_k$，就输出 1。

每一层神经元因为「拥有」上一层神经元的「经验」（输出），所以可以做出更抽象的「决策」。当我们把许多这样的神经元按一定的层次结构连接起来，就得到了人工神经网络（Artificial Neural Network）。

``阿特``：ANN，那我可以叫它安？

``阿扣``：你喜欢咯…… 其实所有的深度学习的神经网络，都可以抽象成三个部分：

![](http://7xjpra.com1.z0.glb.clouddn.com/neuralNetwork1.png)

除了输入和输出层，中间的层都叫隐层。深度神经网络就是隐层数量很多的神经网络，深度学习就是从多层神经网络中，自动学习出各种 pattern。

``阿特``：666！能不能 input 废纸 output 比特币呀？

``阿扣``：……吃药时间到了

### 利用深度神经网络进行学习

``阿扣``：总结一下，对神经网络来说，输入层是数据集/变量，隐层是变量之间的关系（包含变量权重），形成高一级别的「模式」，最后确定输出层的结果。

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a49d8a_hq-perceptron/hq-perceptron.png)

``阿特``：为什么总是说「训练」神经网络好让它「学习」呢？

``阿扣``：训练神经网络的目标，其实就是**计算和调整权重 weights，使得模型输出结果最接近真实的数据集。**

``阿特``：好抽象哦……

``阿扣``：举个例子，我们要预测房价的走势。如果知道房子大小我们可以预测房价，这个关系就可以用一个神经网络节点（node）来简单估计。

![](http://7xjpra.com1.z0.glb.clouddn.com/ngcourseneuro.png)

如果我们知道房子的信息很多的时候怎么办呢？这时候就需要很多的节点，这些节点构成神经网络。房子的多种信息作为输入，房价的预测值作为输出，中间层（可以有多个）是用来计算出前面一层信息的权重，得出一定的模式，传导给下一层，直到最后得出预测值 y。这些中间层可以理解为神经网络。

![](http://7xjpra.com1.z0.glb.clouddn.com/ngcourse_housingprice.png)

``阿特``：好像有点明白了，让机器自己学习中间隐藏起来看不见的「规律」！

``阿扣``：再举个例子，图像识别是深度学习最广泛的应用之一，我们给系统看一张图，它能告诉我们这张图里有没有汪星人：

【翻译这个图】

![](https://pbs.twimg.com/media/CuXT5xEUAAAyXpS.jpg)

``阿特``：哇，原来机器在背后做了这么多事情，我还以为机器都很聪明呢，原来它们只是比较勤奋哈哈哈

``阿扣``：你得到了它~

### Ref

- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Nanodegree | Udacity](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)
- [Neural Networks and Deep Learning | Coursera](https://www.coursera.org/learn/neural-networks-deep-learning)
