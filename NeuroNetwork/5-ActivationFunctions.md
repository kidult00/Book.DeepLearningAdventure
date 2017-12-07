## 激活函数 activation functions

Activation function 激活函数：Activation functions are functions that decide, given the inputs into the node, what should be the node's output?

logistic (often called the sigmoid), tanh, and softmax

### Sigmoid

$$ sigmoid(x)=1/(1+e​^{−x}) $$

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/58800a83_sigmoid/sigmoid.png)

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5893d15c_sigmoids/sigmoids.png)

sigmoid 函数的导数最大值为0.25。这意味着当使用 sigmoid 进行反向传播时，返回网络的 error 将会在每一层收缩至少75％。 对于接近输入层的层，如果有很多层，weights 更新将会很小。

### ReLU

现在更常用 ReLU(rectified linear units) 作为隐藏的激活函数。如果输入 < 0，ReLU 输出 0；如果输入 >0，输出等于输入值。

f(x) = max(x,0)

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58915ae8_relu/relu.png)

ReLU 的缺点是，梯度很多时，ReLU 单元可能大都是 0，产生大量无效的计算。

From [Andrej Karpathy's CS231n course](http://cs231n.github.io/neural-networks-1/#nn):

> Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be “dead” (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue.

```python
# Hidden Layer with ReLU activation function
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)
```

### Softmax

如果我们想将输入归类到多个目标分类中，适合用 Softmax 作为激活函数。

Softmax 函数的作用是把分数转换为概率。每一个输入（logits）对应一个输出，把分数转换为概率，让正确的分类的概率接近 1，其他结果接近 0。 Softmax 函数的输出相当于分类的概率分布。相比 sigmoid，它做了归一化处理。

![](http://7xjpra.com1.z0.glb.clouddn.com/N_softmax.png)


One-Hot Encoding: transforming labels into one-hot encoded vectors

```python
import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer()

# Here the encoder finds the classes and assigns one-hot vectors
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
lb.transform(labels)
>>> array([[1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0]])
```
