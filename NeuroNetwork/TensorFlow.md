# Tensorflow

在 TensorFlow 里，数据不以整数、浮点数、字符串的形式存储。这些数值被封装成叫做 tensor 的对象。constant 是 值不变的 tensor


TensorFlow的 API 是围绕计算图的思想构建的。

![](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580feadb_session/session.png)

TensorFlow 会话(session) 是运行 graph 的环境。session 负责将操作分配给 GPU 和/或 CPU，包括远程机器。


tf.placeholder() 返回一个张量，从传递给 tf.session.run() 函数的数据中获取其值，在会话运行之前正确设置输入。

```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

tensor 进行运算时需要保证类型一致。

```python
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))
```

tf.Variable() 类创建一个带有可以修改的初始值的 tensor，就像一个普通的 Python 变量。这个 tensor 在会话中存储其状态，因此必须手动初始化 tensor 的状态。

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

用正态分布的随机数对权重初始化是一种很好的做法。随机化权重不会让模型在每次训练时都卡在同一个地方。使用 tf.truncated_normal() 函数从正态分布生成随机数。

```python
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))
```

### Save and Restore TensorFlow Models

Saving Variables

```python
import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialize all the Variables
    sess.run(tf.global_variables_initializer())

    # Show the values of weights and bias
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    # Save the model
    saver.save(sess, save_file)
```

Loading Variables

```python
# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, save_file)

    # Show the values of weights and bias
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))
```




## 材料

书

- [深度学习 (豆瓣)](https://book.douban.com/subject/27087503/)
- [机器学习 (豆瓣)](https://book.douban.com/subject/26708119/)
- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)
- [数学之美 （第二版） (豆瓣)](https://book.douban.com/subject/26163454/)
- [Tensorflow：实战Google深度学习框架 (豆瓣)](https://book.douban.com/subject/26976457/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

课程

- [Deep Learning Nanodegree | Udacity](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)
- [Neural Networks and Deep Learning | Coursera](https://www.coursera.org/learn/neural-networks-deep-learning)
- [Stanford University CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

视频

- [Neural networks - 3Blue1Brown - YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

文章
