# DL笔记：用 python 实现梯度下降的算法

回顾上回讲的梯度下降算法，想实现梯度下降，需要不断更新 w：

$$ \Delta w_{ij} = \eta \delta_j x_i $$

具体步骤如下：

- 初始化权重变化率为 0 ：$\Delta w_i = 0$
- 对训练集中的每一个数据：
  + 做正前传播计算：$\hat y=f(\sum_iw_ix_i)$
  + 计算输出单元的 error term：$\delta=(y-\hat y) * f'(\sum_iw_ix_i)$
  + 更新权重变化率：$\Delta w_i= \Delta w_i + \delta x_i$
- 更新权重 $w_i = w_i + \eta \Delta w_i /m$
- 重复 e 次训练 epochs

### 代码实现

1. 初始化权重变化率为 0

```python
del_w = np.zeros(weights.shape)
```
2. 正向传播计算

```pthon
output = sigmoid(np.dot(x, weights))
```
3. 计算输出单元的 error term

```python
error = y - output
error_term = error * output * (1-output)
```
4. 更新权重变化率

```python
del_w += error_term * x
```
5. 更新权重

```python
weights += learnrate * del_w / n_records
```
6. 重复 epochs

```python
for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Calculate the output
        output = sigmoid(np.dot(x, weights))

        # Calculate the error
        error = y - output

        # Calculate the error term
        error_term = error * output * (1-output)

        # Calculate the change in weights for this sample
        # and add it to the total weight change
        del_w += error_term * x

    # Update weights using the learning rate and the average change in weights
    weights += learnrate * del_w / n_records
```

完整代码：

```python
import numpy as np
from data_prep import features, targets, features_test, targets_test

# Defining the sigmoid function for activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# reserve seed
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Calculate the output
        output = sigmoid(np.dot(x, weights))

        # Calculate the error
        error = y - output

        # Calculate the error term
        error_term = error * output * (1-output)

        # Calculate the change in weights for this sample
        # and add it to the total weight change
        del_w += error_term * x

    # Update weights using the learning rate and the average change in weights
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```
