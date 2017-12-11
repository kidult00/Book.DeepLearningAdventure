# Introduction

# Neural Networks


how to build a simple neural network from scratch using Numpy

the algorithms used to train networks such as gradient descent and backpropagation

model evaluation and validation


Ref

- [Perceptrons](https://en.wikipedia.org/wiki/Perceptron)
- Train networks
- [Numpy](http://www.numpy.org/), [Scikit-learn](http://scikit-learn.org/)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Tensorflow](http://tensorflow.org/)


# Convolutional Networks CNN 卷积神经网络

translation invariance 平移不变性

weight sharing

CovNets are neural networks that share their parameters across space

![](http://7xjpra.com1.z0.glb.clouddn.com/ConvNetworkIllustration.png)

CNN 学会识别基本线条和曲线，然后识别图像中的形状和斑点（blobs），然后到更复杂的对象。最后，CNN通过组合更大，更复杂的对象来分类图像。

一个CNN可能有几个层，每个层可能会在对象的层次结构中捕获不同的层次。第一层是层次结构中的最低层，CNN通常将图像的小部分分成简单的形状，如水平线和垂直线以及简单的色块。随后的层往往是层次结构中的更高层次，通常将更复杂的想法分类，如形状（线的组合），最终组合成完整对象。

CNN 的第一步是通过定义好的 filter，将图像拆分成小块。

### One-hot encode

处理 label

have the probability for the correct class be close to 1 and the probabilities for all the others be close to zero.

![](http://7xjpra.com1.z0.glb.clouddn.com/One-Hot_Encoding.png)

![](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377d67_vlcsnap-2016-11-24-15h52m47s438/vlcsnap-2016-11-24-15h52m47s438.png)

The amount by which the filter slides is referred to as the 'stride'.增加步幅 stride 可以减少每层需要观察的 patches 数量，进而减小模型的大小。但是这通常会降低准确性。

Different filters pick up different qualities of a patch. The amount of filters in a convolutional layer is called the filter depth.

How many neurons does each patch connect to?

That’s dependent on our filter depth. If we have a depth of k, we connect each patch of pixels to k neurons in the next layer. This gives us the height of k in the next layer, as shown below.

![](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/5840ffda_filter-depth/filter-depth.png)

但是为什么要把一个 patch 补丁连接到下一层的多个神经元呢？是不是一个神经元不够好？ 多个神经元可以为我们捕获多个有趣的特征。


![](http://7xjpra.com1.z0.glb.clouddn.com/stride_depth_padding.png)

The weights and biases we learn for a given output layer are shared across all patches in a given input layer.

Dimensionality

How can we calculate the number of neurons of each layer in our CNN?

Given:

- 输入层宽为 W 高为 H
- 卷积层 filter size 为 F
- 步长 stride 为 S
- padding 为 P
- filters 数量为 K

下一层的宽: ``W_out =[ (W−F+2P)/S] + 1``.

```
new_height = (input_height - filter_height + 2 * P)/S + 1
```

输出层的高： ``H_out = [(H-F+2P)/S] + 1``.

```
new_width = (input_width - filter_width + 2 * P)/S + 1
```

输出层的深度等于 filters 个数： `` D_out = K``.

输出 volume 为 ``W_out * H_out * D_out``.

Dimensions in tensorflow

```python
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```
Apply convolution networks

```python
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

### 改进 CNN
#### Pooling

- Max Pooling: At every point of on the feature map, look at a small neighborhood around that point and compute the maximum of all the responses around it.

  ![](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/582aac09_max-pooling/max-pooling.png)

  + 好处：参数不会增加，准确率有所提升
  + 缺点：训练成本更高，更多超参数需要调整

  ```python
  def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')
  ```

- Average Pooling: take an average over the window of pixels around a specific location.


#### 1x1 convolutions

![](http://7xjpra.com1.z0.glb.clouddn.com/1x1%20Convolutions.png)

#### inception
![](http://7xjpra.com1.z0.glb.clouddn.com/Inception%20Module.png)

Ref
- [Deep Learning Nanodegree Foundation - Udacity](https://classroom.udacity.com/nanodegrees/nd101-cn/parts/75367b46-2759-4f0e-9692-ad5cd5589c42/modules/29d25480-a513-4925-ae37-e64cc10c1f33/lessons/2fd24529-215c-47b5-a644-2c23650493f6/concepts/30ecc31b-f1b6-49c7-8e67-6757a9a1bb8b)



![](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581a58be_convolution-schematic/convolution-schematic.gif)

```python
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
```

In TensorFlow, strides is an array of 4 elements; the first element in this array indicates the stride for batch and last element indicates stride for features. It's good practice to remove the batches or features you want to skip from the data set rather than use a stride to skip them. You can always set the first and last element to 1 in strides in order to use all batches and features. The middle two elements are the strides for height and width respectively.

To make life easier, the code is using tf.nn.bias_add() to add the bias. Using tf.add() doesn't work when the tensors aren't the same shape.



- http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/
- https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
- http://cs231n.github.io/convolutional-networks/
- http://deeplearning.net/tutorial/lenet.html
- https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
- http://neuralnetworksanddeeplearning.com/chap6.html
- http://xrds.acm.org/blog/2016/06/convolutional-neural-networks-cnns-illustrated-explanation/
- http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
- https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.l6i57z8f2

---

detect and identify objects in images

use convolutional networks to build an autoencoder

pretrained neural network, transfer learning

# Recurrent Neural Networks

well suited to data that forms sequences like text, music, and time series data

word embeddings and implement the Word2Vec model

# Generative Adversarial Networks

### 基础知识

- [Python Programming Course | Udacity](https://www.udacity.com/course/programming-foundations-with-python--ud036)
- [Multivariable Calculus | Khan Academy](https://www.khanacademy.org/math/multivariable-calculus)
- [Linear Algebra | Khan Academy](https://www.khanacademy.org/math/linear-algebra)
- [Learn Data Analysis - Intro to Data Analysis | Udacity](https://www.udacity.com/course/intro-to-data-analysis--ud170)

### Porject

Project 1 - Your First Neural Network
>Build a simple network to make predictions of bike sharing usage.

ref
- [udacity-dlnd-project-1/dlnd-your-first-neural-network.ipynb at master · etakgoz/shape tips](https://github.com/etakgoz/udacity-dlnd-project-1/blob/master/dlnd-your-first-neural-network.ipynb)
- [dlnd-project-1/dlnd-your-first-neural-network.ipynb at master · maihde/graphs](https://github.com/maihde/dlnd-project-1/blob/master/dlnd-your-first-neural-network.ipynb)
- [FlorianWilhelm/udacity-dlnd: Projects of the Udacity Deep Learning Foundation Nanodegree Program](https://github.com/FlorianWilhelm/udacity-dlnd)
- [RyanCCollins/deep-learning-nd: Udacity Deep learning nanodegree projects](https://github.com/RyanCCollins/deep-learning-nd)


Project 2 - Image Classification
>Classify images from the CIFAR-10 dataset using a convolutional neural network.

Project 3 - Generate TV Scripts
>Generate a TV script using a recurrent neural network.

Project 4 - Language Translations
>Use a neural network to translate from one language to another.

Project 5 - GAN Project
>A generative adversarial network (GAN) project.

Project	|Due Date
---|---
First Neural Network	|Dec 7
Image Classification	|Jan 4
Generate TV Scripts	|Jan 25
Translation Project	|Feb 15
Generate Faces	|Mar 8
Final Deadline	|?

### Ref
Books

- [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning](http://www.deeplearningbook.org/)



Articles

- [Convolutional Neural Network](http://neuralnetworksanddeeplearning.com/chap6.html)
- [Recurrent Neural Network](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Generative Adversarial Networks](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks)
- [A Primer on Using LaTeX in Jupyter Notebooks | Democratizing Data](http://data-blog.udacity.com/posts/2016/10/latex-primer/)
