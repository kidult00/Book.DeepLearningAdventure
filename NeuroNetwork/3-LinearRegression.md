# DL笔记：Linear regression 线性回归
``阿扣``：今天带你了解一下线性回归。

``阿特``：🙄 听起来就不是什么容易懂的东西……为什么要了解线……什么，线性回归呢？

``阿扣``：什么[机器学习啊深度学习啊](http://www.uegeek.com/171206DLNote1-ML-DL-Basic.html)，最终目的之一不就是**根据已有数据做出预测**，回归和分类都是「做预测」的主要手段。在下面这张图中找找看，线性回归在机器学习中的位置：

![](http://7xjpra.com1.z0.glb.clouddn.com/LinearRegressionInML.png)

``阿特``：如果说目的都是做「预测」，回归分析和分类有什么不同呢?

``阿扣``：**回归得到预测的具体数值**，比如股市的行情、未来的气温值。**而分类得到一个「声明」，或者说对数据打上的标签**。

``阿特``：那什么是线性回归呢？

``阿扣``：线性回归是最基础的回归类型，它的定义是这样：

> 在统计学中，线性回归（Linear regression）是利用线性回归方程的最小平方函数，对一个或多个自变量和因变量之间关系建模的一种回归分析。这种函数是一个或多个回归系数的模型参数的线性组合。

``阿特``：好吧，看不懂……不过我主要不明白的是「回归」的意思，要回哪里哦……

``阿扣``：初中时学的解方程还记得吧？方程左边有 X，求方程右边的 Y： ax + b =y 。

``阿特``：这个还是记得的。

``阿扣``：回归分析假设 X 和 Y 之间是有奸情哦不对是有关系的，用于了解只有一个自变量 X 时，因变量 Y 的变化。

- 鬼话版：回归分析用来估计模型的参数，以便最好地拟合数据
- 人话版：「回归」的目的呢，就是**找出一个最能够代表所有观测数据的函数，来表示 X 和 Y 的关系**。这个函数只有一个变量，所以是类似这样的一条直线：

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/640px-Linear_regression.svg.png?1512632790654)

``阿特``：好像我记得那种方程在坐标轴上就是用一条直线来表示。不过怎么基于这条直线做预测呢？

``阿扣``：其实不是基于这条线，而是 **「找出」这条最符合 X 和 Y 的关系的线 (line of best fit)，认定这就是它们之间的「关系」，然后去做预测**。

我们先来用符号把这个 X 和 Y 的关系表达式写出来。A 表示我们手上有的数据集，比如你每天的能量摄入和体重值，哈哈哈，然后可以用它来预测你什么时候会变成个胖纸~

``阿特``：紧脏……

``阿扣``：来看看这张图，我告诉你每个字母代表什么：

![](http://7xjpra.com1.z0.glb.clouddn.com/linearClassifier1.png)

``X`` 是每天的能量摄入，``y`` 是体重。我们想预测你的未来体重 $\hat y$ (给字母加个帽子一般表示它的预测值)，于是用 能量输入 乘以一个权重(weight) ``W``，加上一个偏置项(bias) ``b``，就是计算体重的函数了。

$$WX + b = y$$

``阿特``：好像蛮简单的。

``阿扣``：是啊。这个式子以后我们还会无数次看到，是老朋友来的。

关于回归分析，再多说两句。

``阿特``：我有预感不止 20 句……

``阿扣``：它有三个主要用途：

- 因果分析：确定**自变量对因变量的影响的强度**。比如计算剂量和效应，销售和营销支出，年龄和收入之间的关系。
- 预测影响：预测影响或变化的影响，即**因变量随着一个或多个自变量的变化而变化多少**。典型的问题是，「增加一个单位 X， Y 能增加多少？」
- 趋势预测：**预测趋势和未来价值**。比如，「从现在起6个月，黄金的价格是多少？」，「任务 X 的总体成本是多少？」

``阿特``：好像很强大，那它有什么缺点呢？

``阿扣``：有两个主要的缺点：

- 只适用于本身是线性关系的数据
- 对 outliner 敏感

![](http://7xjpra.com1.z0.glb.clouddn.com/lin-reg-w-outliers.png)

比如上图右上角的几个点，偏离平局值比较多，我们叫 outliner。出现这种情况，我们可以试试其他的回归分析类型，或者放弃回归分析，用其他的算法了。

Name|名称|因变量个数|自变量个数
---|---|---|---
Simple linear regression |简单线性回归|1|1
Multiple linear regression |多元线性回归|1|2+
Logistic regression |逻辑回归|1|2+
Ordinal regression |序数回归|1|1+
Multinominal regression |多项式回归|1|1+
Discriminant analysis |判别分析|1|1+

如果需要预测的结果依赖于多个变量，可以用多元线性回归，比如：

$$y = m_1x_1 + m_2x_2 + b$$

我们用一个三维平面来表示这个二元线性回归：

![](http://7xjpra.com1.z0.glb.clouddn.com/just-a-2d-reg.png)

``阿特``：那么多回归类型，不会都要掌握吧？

``阿扣``：嗯，我们接触比较多的是逻辑回归(Logistic regression)。下回给你讲讲逻辑回归要用到的激活函数吧。

``阿特``：🐵 



### Ref

- [迴歸分析 - Wikiwand](https://www.wikiwand.com/zh/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90)
- [線性回歸 - Wikiwand](https://www.wikiwand.com/zh/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8)
- [What is Linear Regression? - Statistics Solutions](http://www.statisticssolutions.com/what-is-linear-regression/)
