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

detect and identify objects in images

use convolutional networks to build an autoencoder

pretrained neural network, transfer learning

# Recurrent Neural Networks

- well suited to data that forms sequences like text, music, and time series data
- word embeddings and implement the Word2Vec model

循环神经网络

先说说 RNN 的用途。RNN 可以从文本中，根据一个字母预测下一个字母应该是什么。这样模型就可以模仿所学习的对象，生成带有这种风格的内容。比如，模仿汪峰写歌。

假设我们要预测单词，"s-t-e-e-p" 当来到第一个 e 时，模型不知道下一个应该输出 e 还是 p，这时 we need to include information about the sequence of characters. we can do this by routing the hidden layer output from the previous step back into the hidden layer.

![](http://7xjpra.com1.z0.glb.clouddn.com/rnn-intro.png)

![](http://7xjpra.com1.z0.glb.clouddn.com/rnn-intro2.png)

$h_t(h_{t-1}W_{hh} )W_{hh}$

![](http://7xjpra.com1.z0.glb.clouddn.com/rnn-intro-layers.png)

RNN 的问题：因为要一层一层传递 w，$y = xw^n$，要么趋于 0 ，要么趋于无限 (vanishing and exploding gradients)。当传递到后面的层时，靠前的层可能已经起不到什么作用了（w 变小，vanishing），就像我们的短时记忆，没法长期保存。这个问题可以用 LSTM (Long Short Term Memory) 解决。

把 recurrent networks 想象成一系列有输入和输出的 cells，每一个 cell 的结构如下：

![](http://7xjpra.com1.z0.glb.clouddn.com/LSTMcell.png)

C 代表 cell state，h 代表 hidden layer。这个 cell 有 4 个 Network layers，就是黄线框的部分。红线框出的是矩阵 element-wise 操作。

第一层是 forget gate。输入经过隐藏的 Sigmoid 变换后，接近 0 的值被 shut off，effectively forgetting that information going forward. 这样网络就可以学习「遗忘」那些导致错误预测的信息，而有用的 long range information 顺利通过。

![](http://7xjpra.com1.z0.glb.clouddn.com/forget_gate.png)

第二层是 update state。更新 cell state

![](http://7xjpra.com1.z0.glb.clouddn.com/rnn-update_state.png)

第三层是隐层输出的 gate。

![](http://7xjpra.com1.z0.glb.clouddn.com/rnn-cstho.png)

把这些 gates 合在一起，LSTM cell consists of a cell state with a bunch of gates used to update it, and leak it out to the hidden state.

为什么 LSTM 能解决梯度消失的问题呢？因为只有 linear sum operation 通过隐层。梯度在网络中传递而不减弱。

LSTM 是 RNN 的基本单元。

Ref

* [Understanding LSTM Networks -- colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [LSTM Networks for Sentiment Analysis — DeepLearning 0.1 documentation](http://deeplearning.net/tutorial/lstm.html)
* [A Beginner's Guide to Recurrent Networks and LSTMs - Deeplearning4j: Open-source, Distributed Deep Learning for the JVM](https://deeplearning4j.org/lstm.html)
* [Recurrent Neural Networks  |  TensorFlow](https://www.tensorflow.org/tutorials/recurrent)
* [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras - Machine Learning Mastery](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)


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
