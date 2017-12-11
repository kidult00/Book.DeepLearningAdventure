# 梯度下降 Gradient Descent

``阿扣``：阿特，还记得训练神经网络的目标其实是什么吗？

``阿特``：我记得好像是要找出最合适的权重(weights)，使得输出结果尽可能接近真实值。

``阿扣``：棒棒！是这样的。为了找到这些权重，我们需要先了解一个重要的方法：梯度下降。

``阿特``：听起来像坐滑滑梯~

``阿扣``：是有那么点意思。说回到训练神经网络，我们需要在训练中及时了解训练效果如何，是不是朝着训练目标在一点点靠近。如果偏离目标，就说明训练模型可能在「犯错」，就要纠正过来。

``阿特``：那怎么知道模型是不是在「犯错」呢？

``阿扣``：我们会找一个度量标准。一个常见的度量是误差的平方和（SSE, sum of the squared errors）：

$$ E=\frac{1}{2}\sum_\mu\sum_j[y^\mu_j - f(\sum_i w_{ij}x^\mu_i)]^2 $$

``阿特``：你……欺负人 >.<

``阿扣``：别着急，我们来拆解这一坨是个什么东西。先看看各个字母的含义：

![](http://7xjpra.com1.z0.glb.clouddn.com/il_for_SSE-1.png)

这个等式里面，有三个求和项，就是这个翻转了 90° 的 M： $\sum$ 。

最右边的求和 $\sum_i w_{ij}x^\mu_i$ 表示我们训练出来的权重，根据输入值 x 得出的目标值 $\hat y$（也就是我们给数据打算的标签），然后用这些结果跟实际的数据中的 y 值做比较看看偏差有多大。

现在你理解了最右边的求和项了吗？

``阿特``：大概意思是我们从数据中预测出来的 y ？

``阿扣``：没错，我们先把这一坨替换成 $\hat y$，简化一下公式：

$$
E=\frac{1}{2}\sum_\mu\sum_j[y^\mu_j - f(\sum_i w_{ij}x^\mu_i)]^2
\\
\downarrow
\\
E=\frac{1}{2}\sum_\mu\sum_j[y^\mu_j - \hat y_j]^2
$$

``阿特``：世界清静多了~

``阿扣``：我们再来看右边这个求和项。j 表示有 j 个隐层节点，把每个节点的误差平方 $[y^\mu_j - \hat y_j]$ 计算出来。最后一个求和项就简单了，它表示把 u 个输出节点的误差加起来。这样就得到了总体误差。

``阿特``：好了，知道了误差，然后呢？

``阿扣``：记得我们的目的是找出能够让误差值（SSE）最小的权重 $w_{ij}$ 。下面有请「梯度下降」 Gradient Descent。

``阿特``：终于能坐滑滑梯了……

``阿扣``：坐这个滑滑梯可能有点晕 😄 。所谓「梯度」其实是多变量函数的导数。

``阿特``：导数？！你说的是微积分里面那个导数吗？ …… 瑟瑟发抖.gif

``阿扣``：别紧张，先听我讲，回忆回忆。

``阿特``：好吧

``阿扣``：你还记得怎么表示函数 f(x) 的导数吧？很简单，就是 f'(x) 。

``阿特``：嗯嗯，记得。

``阿扣``：我们引入「误差项」$\delta$ ，它表示 ``误差 * 激活函数的导数``



![](http://7xjpra.com1.z0.glb.clouddn.com/WX20171127-154242@2x.png)

![](http://img.blog.csdn.net/20170722164242197)

```python
# Defining the sigmoid function for activations
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Input data
x = np.array([0.1, 0.3])
# Target
y = 0.2
# Input to output weights
weights = np.array([-0.8, 0.5])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# the linear combination performed by the node (h in f(h) and f'(h))
h = x[0]*weights[0] + x[1]*weights[1]
# or h = np.dot(x, weights)

# The neural network output (y-hat)
nn_output = sigmoid(h)

# output error (y - y-hat)
error = y - nn_output

# output gradient (f'(h))
output_grad = sigmoid_prime(h)

# error term (lowercase delta)
error_term = error * output_grad

# Gradient descent step
del_w = [ learnrate * error_term * x[0],
          learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x
```

### Stochastic Gradient Descent

Compute the average loss for a very small random fraction of the training data. SGD scales well both data and model size.

for SGD

- inputs: mean = 0, small equal variance
- initial weights: random, mean = 0, small equal variance

如何改善 SGD 效果

- Momentum
  We can take advantage of the knowledge that we've accumulated from previous steps about where we should be headed.
  ![](http://7xjpra.com1.z0.glb.clouddn.com/Momentum.png)
- Learning Rate Decay
  take smaller, noisier steps towards objective. make that step smaller and smaller as you train.
  ![](http://7xjpra.com1.z0.glb.clouddn.com/Learning%20Rate%20Decay.png)
- ADAGRAD

### Mini-batching

小批量对数据集的子集进行训练，而不是一次对所有数据进行训练。这让我们即使在缺乏存储整个数据集的内存时也可以训练。它跟 SGD 结合的效果更好。

在每个 epoch 的开始 shuffle 数据，然后创建小批量。对于每个小批量使用梯度下降来训练网络权重。由于这些批次是随机的，因此每个批次都执行SGD。

```python
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```
None 是 batch size 的 placeholder

### Epochs

epoch 是整个数据集的一个向前和向后传递，用于增加模型的准确性而不需要更多的数据。

- [Gradient Descent with Squared Errors](https://classroom.udacity.com/nanodegrees/nd101-cn/parts/ba124b66-b7f7-43ab-bc89-a390adb57f92/modules/2afd43e6-f4ce-4849-bde6-49d7164da71b/lessons/dc37fa92-75fd-4d41-b23e-9659dde80866/concepts/7d480208-0453-4457-97c3-56c720c23a89)
- [Gradient (video) | Khan Academy](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient)
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#momentum)
