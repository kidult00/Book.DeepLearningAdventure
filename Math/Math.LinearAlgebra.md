- Scalars: 0 dimensional tensor
- Vectors: 1 dimensional tensor

  行向量 $\begin{bmatrix}a \;b \;c   \end{bmatrix}$

  列向量 $\begin{bmatrix}
      1 \\
      2 \\
      3      
    \end{bmatrix}$

- Matrices: 2 dimensional tensor
- Tensors: any n-dimensional collection of values


Element-wise operations: treat items in the matrix individually and perform the same operation on each one

矩阵点乘 [Dot product](https://www.wikiwand.com/en/Dot_product)

Important Reminders About Matrix Multiplication
- The number of columns in the left matrix must equal the number of rows in the right matrix.
- The answer matrix always has the same number of rows as the left matrix and the same number of columns as the right matrix.
- Order matters. Multiplying A•B is not the same as multiplying B•A.
- Data in the left matrix should be arranged as rows, while data in the right matrix should be arranged as columns.

![](http://7xjpra.com1.z0.glb.clouddn.com/dotProduct1.png)

![](http://7xjpra.com1.z0.glb.clouddn.com/Screen%20Shot%202017-11-26%20at%2010.34.17%20AM.png)


you can safely use a transpose in a matrix multiplication if the data in both of your original matrices is arranged as rows
(udacity DLNG lession1-9. Matrix transposes)

consider the transpose just as a different view of your matrix, rather than a different matrix entirely.

- [LINEAR ALGEBRA - khanacademy](https://www.khanacademy.org/math/linear-algebra)
