# Multilayer Neural Networks

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4386_two-layer-network/two-layer-network.png)



### Regularization

[Deep Learning Nanodegree Foundation 3-9 - Udacity](https://classroom.udacity.com/nanodegrees/nd101-cn/parts/75367b46-2759-4f0e-9692-ad5cd5589c42/modules/7dbf2d76-9ec6-494c-b9ce-34ea968347cf/lessons/bd73c076-7661-4947-9f73-7020d313eb6f/concepts/7e05821d-98fd-46be-aefd-74313e7616b6)

#### L2 Regularization

![](http://7xjpra.com1.z0.glb.clouddn.com/L2_Regularization.png)

#### Dropout

Dropout is a regularization technique for reducing overfitting. The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections. Figure 1 illustrates how dropout works.

![](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58222112_dropout-node/dropout-node.jpeg)

```python
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```

``keep_prob`` allows you to adjust the number of units to drop. In order to compensate for dropped units, ``tf.nn.dropout()`` multiplies all units that are kept (i.e. not dropped) by ``1/keep_prob``.
