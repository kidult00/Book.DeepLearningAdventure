My tribe: Data Analyst interested in Better Explaining Data

## 什么是机器学习？
[What is Machine Learning: A Tour of Authoritative Definitions and a Handy One-Liner You Can Use - Machine Learning Mastery](http://machinelearningmastery.com/what-is-machine-learning/)

大学教材中相对权威的定义：

1. Tom Mitchell in his book [Machine Learning](http://www.amazon.com/dp/0070428077?tag=inspiredalgor-20)

	> The field of machine learning is concerned with the question of how to construct computer programs that automatically improve with experience.
	> 
	> 机器学习聚焦于一个问题：如何构建随着经验而自动改进的计算机程序。
	
	关键词：会自我改进的程序
	
	- E：experience , what data to collect
	- T：tasks , what decisions the software needs to make
	- P：performance measure , how we will evaluate it’s results 

2. [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](http://www.amazon.com/dp/0387848576?tag=inspiredalgor-20)

	> Vast amounts of data are being generated in many fields, and the statisticians’s job is to make sense of it all: to extract important patterns and trends, and to understand “what the data says”. We call this learning from data.
	> 
	> 从数据中提取重要的模式和规律/趋势

3. Bishop in the preface of his book [Pattern Recognition and Machine Learning](http://www.amazon.com/dp/0387310738?tag=inspiredalgor-20) comments:

	>Pattern recognition has its origins in engineering, whereas machine learning grew out of computer science. However, these activities can be viewed as two facets of the same field…
	>
	>模式识别和机器学习是一体两面

4. Marsland provides adopts the Mitchell definition of [Machine Learning: An Algorithmic Perspective](http://www.amazon.com/dp/B005H6YE18?tag=inspiredalgor-20)

	> One of the most interesting features of machine learning is that it lies on the boundary of several different academic disciplines, principally computer science, statistics, mathematics, and engineering. …machine learning is usually studied as part of artificial intelligence, which puts it firmly into computer science …understanding why these algorithms work requires a certain amount of statistical and mathematical sophistication that is often missing from computer science undergraduates.
	> 
	> 机器学习基于计算机、统计学、数学和工程。

5. Drew Conway created a nice Venn Diagram in September 2010 that might help. 

	In his explanation he comments Machine Learning is Hacking + Math & Statistics. 
	
	![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Data_Science_VD.png)

6. [Jason Brownlee](http://machinelearningmastery.com/author/jasonb/) in [What is Machine Learning: A Tour of Authoritative Definitions and a Handy One-Liner You Can Use - Machine Learning Mastery](http://machinelearningmastery.com/what-is-machine-learning/)

	> Machine Learning is the training of a model from data that generalizes a decision against a performance measure.
	> 
	> 机器学习是通过用于决策的数据去训练模型，并达到某些运行标准

### 机器学习的应用

[Practical Machine Learning Problems - Machine Learning Mastery](http://machinelearningmastery.com/practical-machine-learning-problems/)

1.  Spam Detection 识别垃圾邮件
	- E：邮件
	- T：决策问题（分类），把每一封邮件标记为垃圾或正常邮件
	- P：准确率
	
	在这里：
	
	- 准备决策程序 -- 训练
	- 收集的数据 -- 训练集
	- 程序 -- 模型
2. Credit Card Fraud Detection
3. Digit Recognition：手写数字识别
4. Speech Understanding
5. Face Detection
6. Product Recommendation
7. Medical Diagnosis：匹配病人症状和数据库中的症状，预测是否可能患病
8. Stock Trading
9. Customer Segmentation：对比所有用户历史行为记录，和当前用户行为模式，判断用户类型
10. Shape Detection

### 机器学习的问题类型

- Classification 分类，如垃圾邮件识别
- Regression 回归，例如股市预测
- Clustering 聚类，如 iPhoto 按人分组
- Rule Extraction 规则提取，如数据挖掘

### 预测模型是应用机器学习最有用的部分
Predictive Modeling

有监督学习 supervised learning

给定结果和关系，构建模型去做识别和判断。例子：[iris exercise](http://machinelearningmastery.com/how-to-run-your-first-classifier-in-weka/)，构建一个模型，通过花的一些测量数据，就可以判断这个花属于什么种类。如果输出是类别 category，则问题分类 classification 问题；如果输出是数值，，则属于回归 regression 问题。

> The algorithm does the learning. The model contains the learned relationships.

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2015/09/Create-a-Predictive-Model.png)

用模型进行预测

1. Sample Data: the data that we collect that describes our problem with known relationships between inputs and outputs.
2. Learn a Model: the algorithm that we use on the sample data to create a model that we can later use over and over again.
3. Making Predictions: the use of our learned model on new data for which we don’t know the output.

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2015/09/Make-Predictions.png)

出处：[Gentle Introduction to Predictive Modeling](http://machinelearningmastery.com/gentle-introduction-to-predictive-modeling/)

## 机器学习算法概览

> model = algorithm(data)

[A Tour of Machine Learning Algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

###  一、按 learning style 分

![](https://s3.amazonaws.com/MLMastery/MachineLearningAlgorithms.png)

#### Supervised Learning 监督学习

分类结果/标签已知。预测错误时则改进模型，直到达到某个水平。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Supervised-Learning-Algorithms.png)

典型问题是分类和回归，典型算法包括  Logistic Regression 和 the Back Propagation Neural Network.

#### Unsupervised Learning 非监督学习
结果未知

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Unsupervised-Learning-Algorithms.png)

典型问题包括聚类，降维 dimensionality reduction 和 关联规则学习 association rule learning.

典型算法包括 关联规则 the Apriori algorithm 和 k-Means.


#### Semi-Supervised Learning 半监督学习

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Semi-supervised-Learning-Algorithms.png)

There is a desired prediction problem but the model must learn the structures to organize the data as well as make predictions.

典型问题是分类和回归

### 二、按 功能相似性 分

#### Regression Algorithms 回归算法
回归对变量间反复出现的关系建模

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Regression-Algorithms.png)

* Ordinary Least Squares Regression (OLSR)
* Linear Regression
* Logistic Regression
* Stepwise Regression
* Multivariate Adaptive Regression Splines (MARS)
* Locally Estimated Scatterplot Smoothing (LOESS)

#### Instance-based Algorithms / winner-take-all methods / memory-based learning

对模型来说重要的训练数据实例的决策问题 Instance-based learning model is a decision problem with instances or examples of training data that are deemed important or required to the model. Focus is put on the representation of the stored instances and similarity measures used between instances.

建立样本数据集并且用 similarity measure 跟新数据比对，找出最佳匹配然后做出预测。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Instance-based-Algorithms.png)

* k-Nearest Neighbor (kNN)
* Learning Vector Quantization (LVQ)
* Self-Organizing Map (SOM)
* Locally Weighted Learning (LWL)

#### Regularization Algorithms 正则化算法

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Regularization-Algorithms.png)

* Ridge Regression
* Least Absolute Shrinkage and Selection Operator (LASSO)
* Elastic Net
* Least-Angle Regression (LARS)


#### Decision Tree Algorithms

基于数据属性值构建决策模型。Decisions fork in tree structures until a prediction decision is made for a given record. 

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Decision-Tree-Algorithms.png)

* Classification and Regression Tree (CART)
* Iterative Dichotomiser 3 (ID3)
* C4.5 and C5.0 (different versions of a powerful approach)
* Chi-squared Automatic Interaction Detection (CHAID)
* Decision Stump
* M5
* Conditional Decision Trees

#### Bayesian Algorithms

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Bayesian-Algorithms.png)

* Naive Bayes
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Averaged One-Dependence Estimators (AODE)
* Bayesian Belief Network (BBN)
* Bayesian Network (BN)

#### Clustering Algorithms

describes the class of problem and the class of methods. All methods are concerned with using the inherent structures in the data to best organize the data into groups of maximum commonality.

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Clustering-Algorithms.png)

* k-Means
* k-Medians
* Expectation Maximisation (EM)
* Hierarchical Clustering

#### Association Rule Learning Algorithms 关联规则学习算法

从变量关系中提取最佳解释规则

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Assoication-Rule-Learning-Algorithms.png)

* Apriori algorithm
* Eclat algorithm

#### Artificial Neural Network Algorithms 人工神经网络算法

受生物神经网络结构和功能启发的算法

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Artificial-Neural-Network-Algorithms.png)

* Perceptron
* Back-Propagation
* Hopfield Network
* Radial Basis Function Network (RBFN)


#### Deep Learning Algorithms 深度学习算法

a modern update to Artificial Neural Networks that **exploit abundant cheap computation**.  many methods are concerned with semi-supervised learning problems where large datasets contain very little labeled data.

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Deep-Learning-Algorithms.png)

* Deep Boltzmann Machine (DBM)
* Deep Belief Networks (DBN)
* Convolutional Neural Network (CNN)
* Stacked Auto-Encoders


#### Dimensionality Reduction Algorithms 降维算法

跟聚类方法类似，降维算法提取数据的内在结构，但更多以非监督方式进行。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Dimensional-Reduction-Algorithms.png)

* Principal Component Analysis (PCA)
* Principal Component Regression (PCR)
* Partial Least Squares Regression (PLSR)
* Sammon Mapping
* Multidimensional Scaling (MDS)
* Projection Pursuit
* Linear Discriminant Analysis (LDA)
* Mixture Discriminant Analysis (MDA)
* Quadratic Discriminant Analysis (QDA)
* Flexible Discriminant Analysis (FDA)

#### Ensemble Algorithms 整体算法

组合多个分别训练的次优模型做出总体预测

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Ensemble-Algorithms.png)

* Boosting
* Bootstrapped Aggregation (Bagging)
* AdaBoost
* Stacked Generalization (blending)
* Gradient Boosting Machines (GBM)
* Gradient Boosted Regression Trees (GBRT)
* Random Forest

#### 其他算法

* Feature selection algorithms
* Algorithm accuracy evaluation
* Performance measures
* Computational intelligence (evolutionary algorithms, etc.)
* Computer Vision (CV)
* Natural Language Processing (NLP)
* Recommender Systems
* Reinforcement Learning
* Graphical Models

其他 list

* [List of Machine Learning Algorithms](http://en.wikipedia.org/wiki/List_of_machine_learning_algorithms): On Wikipedia. Although extensive, I do not find this list or the organization of the algorithms particularly useful.
* [Machine Learning Algorithms Category](http://en.wikipedia.org/wiki/Category:Machine_learning_algorithms): Also on Wikipedia, slightly more useful than Wikipedias great list above. It organizes algorithms alphabetically.
* [CRAN Task View: Machine Learning & Statistical Learning](http://cran.r-project.org/web/views/MachineLearning.html): A list of all the packages and all the algorithms supported by each machine learning package in R. Gives you a grounded feeling of what’s out there and what people are using for analysis day-to-day.
* [Top 10 Algorithms in Data Mining](http://www.cs.uvm.edu/~icdm/algorithms/index.shtml): [Published article](http://link.springer.com/article/10.1007/s10115-007-0114-2) and now a [book](http://www.amazon.com/dp/1420089641?tag=inspiredalgor-20) on the most popular algorithms for data mining. Another grounded and less overwhelming take on methods that you could go off and learn deeply.

如何学习算法

* [How to Learn Any Machine Learning Algorithm](http://machinelearningmastery.com/how-to-learn-a-machine-learning-algorithm/): A systematic approach that you can use to study and understand any machine learning algorithm using “algorithm description templates”
* [How to Create Targeted Lists of Machine Learning Algorithms](http://machinelearningmastery.com/create-lists-of-machine-learning-algorithms/): How you can create your own systematic lists of machine learning algorithms to jump start work on your next machine learning problem.
* [How to Research a Machine Learning Algorithm](http://machinelearningmastery.com/how-to-research-a-machine-learning-algorithm/): A systematic approach that you can use to research machine learning algorithms.
* [How to Investigate Machine Learning Algorithm Behavior](http://machinelearningmastery.com/how-to-investigate-machine-learning-algorithm-behavior/): A methodology you can use to understand how machine learning algorithms work by creating and executing very small studies into their behavior.
* [How to Implement a Machine Learning Algorithm](http://machinelearningmastery.com/how-to-implement-a-machine-learning-algorithm/): A process and tips and tricks for implementing machine learning algorithms from scratch.



## 什么是深度学习？

[神经网络与深度学习](https://www.gitbook.com/book/tigerneil/neural-networks-and-deep-learning-zh)

深度学习：一个技术集合，用于神经网络学习

## 什么是神经网络？

神经网络：受生物学启发的**编程范式**，让计算机从观测数据中学习

手写识别常常被当成学习神经网络的原型问题。

神经网络的思想是获取大量训练样本，然后开发出一个可以从样本中学习（推断出规则）的系统。

## 什么是反向传播算法
反向传播算法最初在 1970 年代被发现，但是这个算法的重要性直到 David Rumelhart、Geoffrey Hinton 和 Ronald Williams 的 1986年的论文中才被真正认可。现在，反向传播算法已经是神经网络学习的重要组成部分了。

## 从入门到擅长：五步掌握机器学习应用方法
[从入门到擅长：五步掌握机器学习应用方法](file:///Users/kidult/OneDrive/8-Material/Software/zotero/storage/6ANXTDIA/1656.html)

![](file:///Users/kidult/OneDrive/8-Material/Software/zotero/storage/6ANXTDIA/5805cad0d17df.png)

将这条路径总结为5个步骤：

- 步骤1：调整心态（相信！）
- 步骤2：挑选一个过程（怎样得到结果）
- 步骤3：挑选一种工具（实施）
- 步骤4：在数据集上应用（投入工作）
- 步骤5：建立一个组合（展现你的技能）

### 步骤0：界标
机器学习的分界范围