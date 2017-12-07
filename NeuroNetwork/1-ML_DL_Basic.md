## DL笔记：机器学习和深度学习的区别

![](https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg.png)
via [The Difference Between AI, Machine Learning, and Deep Learning? | NVIDIA Blog](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/)

Nvidia 博客上的这张图很好表示了 AI, Machine Learning, Deep Learning 三者的关系。人工智能是一类非常广泛的问题，机器学习是其中一个重要领域和手段，**深度学习则是机器学习的一个分支**。在很多人工智能问题上，深度学习的方法突破了传统机器学习的瓶颈，因而影响力迅速扩大。

### 什么是机器学习？

![](https://uploads.toptal.io/blog/image/443/toptal-blog-image-1407508081138.png)

00 试着翻出一些机器学习相对权威的定义，看看它们有什么共同点：

Definition|Translation|Source|Key words
---|---|---|---
The field of machine learning is concerned with the question of how to construct computer programs that automatically improve with experience.|机器学习聚焦于一个问题：如何构建随着经验而自动改进的计算机程序。|Tom Mitchell in  [Machine Learning](http://www.amazon.com/dp/0070428077?tag=inspiredalgor-20)|会自我改进的程序
Vast amounts of data are being generated in many fields, and the statisticians’s job is to make sense of it all: to extract important patterns and trends, and to understand “what the data says”. We call this learning from data.|从数据中提取重要的模式和规律/趋势|[The Elements of Statistical Learning: Data Mining, Inference, and Prediction](http://www.amazon.com/dp/0387848576?tag=inspiredalgor-20)|模式提取
Pattern recognition has its origins in engineering, whereas machine learning grew out of computer science. However, these activities can be viewed as two facets of the same field…|模式识别和机器学习是一体两面|Bishop in [Pattern Recognition and Machine Learning](http://www.amazon.com/dp/0387310738?tag=inspiredalgor-20)|模式识别
Machine Learning is the training of a model from data that generalizes a decision against a performance measure.|机器学习是通过用于决策的数据去训练模型，并达到某些运行标准|[Jason Brownlee](http://machinelearningmastery.com/author/jasonb/) in [What is Machine Learning: A Tour of Authoritative Definitions and a Handy One-Liner You Can Use](http://machinelearningmastery.com/what-is-machine-learning/)|通过数据训练模型

简单来说，就是机器通过一系列「任务」从「经验」（数据）中学习，并且评估「效果」如何：

![](http://7xjpra.com1.z0.glb.clouddn.com/Col.DL.ETP.png)

为什么叫做「学习」呢？一般编程语言的做法，是定义每一步指令，逐一执行并最终达到目标。而机器学习则相反，先定义好输出，然后程序自动「学习」出达到目标的「步骤」。

机器学习可以分为：

- 监督学习：给出定义好的标签，程序「学习」标签和数据之间的映射关系
- 非监督学习：没有标签的数据集
- 强化学习：达到目标会有正向反馈

![](https://i1.wp.com/cybrml.com/wp-content/uploads/2017/01/MachineLearningDiagram.png?resize=770%2C551)

### 机器学习擅长做什么？

当然是替代重复的人工劳动，用机器自动从大量数据中识别模式——也就是「套路」啦。知道「套路」后，我们可以干嘛呢？

- Classification 分类，如垃圾邮件识别(detection, ranking)
- Regression 回归，例如股市预测
- Clustering 聚类，如 iPhoto 按人分组
- Rule Extraction 规则提取，如数据挖掘

比如垃圾邮件识别的问题，做法是先从每一封邮件中抽取出对识别结果可能有影响的因素（称为特征 feature），比如发件地址、邮件标题、收件人数量等等。然后使用算法去训练数据中每个特征和预测结果的相关度，最终得到可以预测结果的特征。

算法再强大，如果无法从数据中「学习到」更好的特征表达，也是徒劳。同样的数据，使用不同的表达方法，可能会极大影响问题的难度。一旦解决了数据表达和特征提取问题，很多人工智能任务也就迎刃而解。

### 为什么需要深度学习？

但是对机器学习来说，特征提取并不简单。特征工程往往需要人工投入大量时间去研究和调整，就好像原本应该机器解决的问题，却需要人一直在旁边搀扶。

深度学习便是解决特征提取问题的一个机器学习分支。它可以自动学习特征和任务之间的关联，还能从简单特征中提取复杂的特征。

![](http://7xjpra.com1.z0.glb.clouddn.com/Col.DL.ML_vs_DL.png)




### Ref
- [What is Machine Learning: A Tour of Authoritative Definitions and a Handy One-Liner You Can Use - Machine Learning Mastery](http://machinelearningmastery.com/what-is-machine-learning/)
- [机器学习 (豆瓣)](https://book.douban.com/subject/26708119/)
- [Tensorflow：实战Google深度学习框架 (豆瓣)](https://book.douban.com/subject/26976457/)
