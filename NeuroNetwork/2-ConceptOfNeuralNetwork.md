

## 神经网络



在生物神经网络中，每个神经元与其他神经元相连，当它接收外界信号（其他神经元）作为输入，当电位超过了某个阈值（threshold）而被「激活」时，会向相连的神经元「发射」（fire）信号。

![](http://7xjpra.com1.z0.glb.clouddn.com/neuralNetwork.png)

### Perceptrons 感知器

在计算机科学中，我们将独立的计算单元看做神经元。

比如说，我们把神经元看做包含一个 0 到 1 之间数字的小球：

0.8

神经元里面的数字就是 activation。当数字超过某个阈值，比如说 0.5 时，我们就定义这个神经元被激活了，它会输出 1 作为信号。如果神经元包含的数字小于 0.5，那它就输出 0，表示没有被激活。



我们可以把上面提到的神经元成为 Perceptrons 感知器。一个感知器接收若干二级制输入 $x_1,x_2,...$，然后产生一个二进制输出：

![](http://neuralnetworksanddeeplearning.com/images/tikz0.png)

这个最简单的系统里，包含：

- 输入：这个神经元接收到的其他神经元的信号
- 判断器：激活函数
- 输出：1 表示 yes「发射」，0 表示 no「不发射」

![](https://viniciusarruda.github.io/images/mp_neuron.png)

怎么计算输出呢？我们引入「权重」weights，它表示从输入到输出的重要程度。权重的和 $\sum_j w_jx_j$ 如果大于阈值 $v_k$，就输出 1。

每一层神经元因为「拥有」上一层神经元的「经验」（输出），所以可以做出更抽象的「决策」。当我们把许多这样的神经元按一定的层次结构连接起来，就得到了人工神经网络（Artificial Neural Network）。

![](http://7xjpra.com1.z0.glb.clouddn.com/neuralNetwork1.png)

除了输入和输出层，中间的层都叫隐层。深度神经网络就是隐层数量很多的神经网络，深度学习就是从多层神经网络中，自动学习出各种 pattern。

### 深度神经网络学习



举个例子，我们要预测房价的走势。如果知道房子大小我们可以预测房价，这个关系就可以用一个神经网络节点（node）来简单估计。


![](http://7xjpra.com1.z0.glb.clouddn.com/ngcourseneuro.png)


如果我们知道房子的信息很多的时候怎么办呢？这时候就需要很多的节点，这些节点构成神经网络。房子的多种信息作为输入，房价的预测值作为输出，中间层（可以有多个）是用来计算出前面一层信息的权重，得出一定的模式，传导给下一层，直到最后得出预测值 y。这些中间层可以理解为神经网络。

![](http://7xjpra.com1.z0.glb.clouddn.com/ngcourse_housingprice.png)

【翻译这个图】

![](https://pbs.twimg.com/media/CuXT5xEUAAAyXpS.jpg)


### Ref

- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Nanodegree | Udacity](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)
- [Neural Networks and Deep Learning | Coursera](https://www.coursera.org/learn/neural-networks-deep-learning)
