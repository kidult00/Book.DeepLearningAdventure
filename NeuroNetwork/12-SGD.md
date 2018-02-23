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
