### Data dimensions 数据维度

按照维度，可以把数据分为几类：

- Scalars: 0 维的单个数值，如 100，如 -0.3
- Vectors: 1 维的一组数值，如 [1 2 3]

  行向量 $\begin{bmatrix}a \;b \;c   \end{bmatrix}$

  列向量 $\begin{bmatrix}
      1 \\
      2 \\
      3      
    \end{bmatrix}$

- Matrices: 2 维的数值矩阵，比如下面这个 2行3列 的矩阵

  $\begin{bmatrix}
      1 \;2 \; 3 \\
      4 \;5 \; 6 \\      
    \end{bmatrix}$

- Tensors: 任意维度 （Scalars 是 0 维 Tensor，Vectors 是 1 维的 Tensor，Matrices 是 2 维的 Tensor）


### 矩阵点乘

Element-wise operations 对矩阵内的每一个元素都执行运算。比如点乘 [Dot product](https://www.wikiwand.com/en/Dot_product)。

![](http://7xjpra.com1.z0.glb.clouddn.com/IMG.Math.dot_product.png)

矩阵乘法的要点：

- 左边矩阵的列数，必需与右边矩阵的行数相同
- 结果矩阵的行数与左边矩阵相同，列数与右边矩阵相同
- 相乘顺序会影响结果，A•B ≠ B•A
- **左边矩阵的数据应该按行组织，右边矩阵的数据应该按列组织**

![](http://7xjpra.com1.z0.glb.clouddn.com/Screen%20Shot%202017-11-26%20at%2010.34.17%20AM.png)

如果数据都是按行排列的，就可以转置矩阵，结果不变。
(via Udacity DLNG lession1-9. Matrix transposes)

矩阵转置实际上是变换了矩阵的 view，而没有改变矩阵本身。

Ref

- [Deep Learning Nanodegree | Udacity](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)
- [LINEAR ALGEBRA - khanacademy](https://www.khanacademy.org/math/linear-algebra)
