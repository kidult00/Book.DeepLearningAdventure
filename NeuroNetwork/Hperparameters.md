# 超参数

没有通用的超参数。最佳参数要根据 task 和 dataset 来设定。

超参数可以分为两类

- optimizer hyperparameters: learning rate, minibatch size, epochs
- model hyperparameters: number of layers and hidden units, like RNN specific parameters

## Learning Rate

 > The single most important hyperparameter and one should always make sure that is has been tuned.
 > --- Yoshua Bengio

 Good starting point = 0.01

 ![](http://7xjpra.com1.z0.glb.clouddn.com/IMG.DL.LearningRate1.png)


 Learning Rate Decay


## Minibatch size

Good starting point: 32,64,128,256

A larger minibatch size allows computational boosts that utilizes matrix multiplication, in the training calculations. But needs more momery.

太小的话，训练会很慢。太大的话，需要更多的计算资源，也会影响准确率。


## Number of Iterations

Early stopping: 监测 validation error，如果不再下降，就停止训练。

More recent versions of TensorFlow deprecated monitors in favor of [SessionRunHooks](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook). SessionRunHooks are an evolving part of tf.train, and going forward appear to be the proper place where you'd implement early stopping.

Two pre-defined stopping monitors exist as a part of tf.train's training hooks:

- [StopAtStepHook](https://www.tensorflow.org/api_docs/python/tf/train/StopAtStepHook): A monitor to request the training stop after a certain number of steps
- [NanTensorHook](https://www.tensorflow.org/api_docs/python/tf/train/NanTensorHook): a monitor that monitor's loss and stops training if it encounters a NaN loss


## Number of Hidden Units/layers

如果太大，会过拟合。

Regularization techniques: Dropout, L2

- validation error 增大之前，增加 hidden units
- 第一个隐藏节点数，应该比 input 节点数多

>"in practice it is often the case that 3-layer neural networks will outperform 2-layer nets, but going even deeper (4,5,6-layer) rarely helps much more. This is in stark contrast to Convolutional Networks, where depth has been found to be an extremely important component for a good recognition system (e.g. on order of 10 learnable layers)." ~ Andrej Karpathy in https://cs231n.github.io/neural-networks-1/


some tasks show reasonable performance with embedding sizes between 50-200


### Ref
- [tf.train.exponential_decay  |  TensorFlow](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)
- [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228)
- [SessionRunHooks](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook)
- [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) by Yoshua Bengio
- [Deep Learning book - chapter 11.4: Selecting Hyperparameters](http://www.deeplearningbook.org/contents/guidelines.html) by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- [Neural Networks and Deep Learning book - Chapter 3: How to choose a neural network's hyper-parameters?](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters) by Michael Nielsen
- [Efficient BackProp (pdf)](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCun
- [How to Generate a Good Word Embedding?](https://arxiv.org/abs/1507.05523) by Siwei Lai, Kang Liu, Liheng Xu, Jun Zhao
- [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228) by Dmytro Mishkin, Nikolay Sergievskiy, Jiri Matas
- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078) by Andrej Karpathy, Justin Johnson, Li Fei-Fei
