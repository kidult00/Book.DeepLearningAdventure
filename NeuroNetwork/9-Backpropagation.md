## Backpropagation 反向传播算法

反向传播算法是深度学习的核心之一。

对多层神经网络，我们仍然用梯度下降法去训练。之前我们已经学过如何计算输出节点的误差项 $\delta =(y-\hat y)f'(h)$，借助梯度下降算法用误差项训练隐层到输出层的权重。

那现在还有输入层到隐藏层的权重，该怎么求隐层节点对应的误差项呢？在神经网络里，输出节点的误差项，跟隐层 的权重是成比例的

![](http://7xjpra.com1.z0.glb.clouddn.com/vlcsnap-2017-12-19-15h02m16s033.png)

既然我们知道输出的误差项，那么可以用它来「反向传播」，求出隐层的误差项，再用于求输入节点的权重。

比如，输出层 k 个节点对应的误差项是 $\delta^o_k$ 。隐层有 j 个节点，那么隐层节点到输出节点的 j 个误差项是：

![](http://7xjpra.com1.z0.glb.clouddn.com/backprop-error.gif)

![](http://7xjpra.com1.z0.glb.clouddn.com/vlcsnap-2017-12-19-15h06m32s756.png)

（$\delta_k$ 成为了 wx + b 中的变量 x）

跟正向传播类似，权重的更新为 $\Delta w_{ij}=\eta \delta^h_jx_i$ 。

现在我们考虑最简单的神经网络：只有一个隐藏节点，只有一个输出节点。用反向传播算法更新权重的算法如下：

- 给每一次的权重赋值为 0
  + 输入层→隐层的权重 $\Delta w_{ij}=0$
  + 隐层→输出层的权重 $\Delta W_j=0$
​
- 对训练数据中的每一个数据：
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


### 用 Python 实现反向传播算法

```python
import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):

      ## Forward pass ##

        # Calculate the output
        hidden_input = np.dot(x, weights_input_hidden) # x·w
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))

      ## Backward pass ##

        # Calculate the network's prediction error
        error = y - output

        # Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:,None] # x.T

    # Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

```

Ref

- [Yes you should understand backprop – Medium](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
- [CS231n Winter 2016 Lecture 4 Backpropagation, Neural Networks 1-Q_UWHTY_TEQ.mp4 - YouTube](https://www.youtube.com/watch?v=59Hbtz7XgjM)
