# 菜鸟数据科学入门04 - NumPy通用函数和基本数据处理



## 通用函数
通用函数（ufunc）是对 ndarray 中的数据执行元素级运算的函数。NumPy 中的通用函数可以看做矢量化的包装器。

### 一元 ufunc

函数|说明
---|---
abs, fabs|计算整数、浮点数或复数的绝对值，非复数可以用 fabs
sqrt|计算各元素的平方根，相当于 arr ** 0.5
square|计算各元素的平方，相当于 arr ** 2
exp|计算各元素的指数 $$ e^x $$
log, log10, log2, log1p|分别为自然对数（底为 e）、底数为10的 log、底数为2的 log、log（1+x）
sign|计算各元素的正负号: 1 (正数), 0 (零), -1 (负数)
ceil|计算各元素的 ceiling 值, 即大于等于该值的最小整数
floor|计算各元素的 floor 值, 即小于等于该值的最大整数
rint|将各元素值四舍五入到最接近的整数，保留 dtype
modf|将数组的小数和整数部分以两个独立数组的形式返回
cos, cosh, sin, sinh, tan, tanh|普通型和双曲型三角函数
arccos, arccosh, arcsin, arcsinh, arctan, arctanh|反三角函数
logical_not|计算各元素 not x 的真值。相当于 -arr

比如 rint 函数：

```python
In[]:
arr = np.linspace(1,10,20)
print(arr)
np.rint(arr)

[  1.           1.47368421   1.94736842   2.42105263   2.89473684
   3.36842105   3.84210526   4.31578947   4.78947368   5.26315789
   5.73684211   6.21052632   6.68421053   7.15789474   7.63157895
   8.10526316   8.57894737   9.05263158   9.52631579  10.        ]
   
Out[]:
array([  1.,   1.,   2.,   2.,   3.,   3.,   4.,   4.,   5.,   5.,   6.,
         6.,   7.,   7.,   8.,   8.,   9.,   9.,  10.,  10.])
```

### 二元 ufunc

函数|说明
---|---
add|数组中对应元素相加
subtract|从第一个数组中减去第二个数组中的元素
multiply|数组元素相乘
divide, floor_divide|除法或向下去余除法
power|对第一个数组中的元素 A，根据第二个数组中相应元素 B，计算$$ A^B $$
maximum, fmax|元素级的最大值计算，fmax 将忽略 NaN
minimum, fmin|元素级的最小值计算，fmin 将忽略 NaN
mod|元素级求模（除法取余）
copysign|将第二个数组中值的符号复制给第一个数组中的值
greater, greater_equal, less, less_equal, equal, not_equal|比较运算，产生布尔型数组，相当于 >, >=, <, <=, ==, !=
logical_and, logical_or, logical_xor|元素级真值逻辑运算，相当于 &, \|, ^

## 利用数组进行数据处理
NumPy 数组可以将多种数据处理任务表述为简洁的数组表达式。**用数组代替循环的做法，通常称为矢量化。**

### 1.将条件逻辑表述为数组运算
``numpy.where`` 函数是三元表达式 ``x if condition else y`` 的矢量化版本。

假设我们有两个数值数组和一个布尔数组：

```python
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
```
如果我们想做到：当 cond 中的值为 True 时取 xarr 的值，否则取 yarr 的值，用列表推倒式的写法会是：

```python
In []: 
result = [(x if c else y)
		for x, y, c in zip(xarr, yarr, cond)]
result

Out[]: 
[1.1000000000000001, 2.2000000000000002, 1.3, 1.3999999999999999, 2.5]
```
但是这样做的话，处理速度有限，而且无法用于多维数组。如果使用 numpy.where，则可以写得非常简洁：

```python
In []: result = np.where(cond, xarr, yarr)
	   result
Out[]: array([ 1.1,  2.2,  1.3,  1.4,  2.5])
```
在数据分析工作中，``where`` 通常用于根据一个数组而产生另一个新的数组。

其他常用的逻辑函数：

函数|说明
---|---
np.choose|根据给定的索引数组，从列表中选择值np.select|根据条件的数组，从列表中选择值np.nonzero|返回一个非零元素的数组索引
### 2.基础统计

函数|说明
---|---
sum|对数组中全部或某轴向的元素求和
mean|均值
std, var|标准差，方差
min, max|最大值，最小值
argmin, argmax|	最大和最小元素的索引
cumsum|	所有元素的累计和
cumprod|所有元素的累计积

用 ``sum`` 对不同轴求和：

![](http://7xjpra.com1.z0.glb.clouddn.com/170105array_aggregation.png)

其他函数示例：

```python
In []: arr = np.random.randn(5, 4) # normally-distributed data
	   arr.mean()
Out[]: 0.062814911084854597

In []: np.mean(arr)
Out[]: 0.062814911084854597

In []: arr.mean(axis=1)
Out[]: array([-1.2833,  0.2844,  0.6574,  0.6743, -0.0187])
```



### 3.排序
用 ``sort``函数排序。多维数组可以在任何一个轴向上排序：

```python
In []: arr = randn(5, 3)
	   arr
Out[]: 
array([[-0.7139, -1.6331, -0.4959],
       [ 0.8236, -1.3132, -0.1935],
       [-1.6748,  3.0336, -0.863 ],
       [-0.3161,  0.5362, -2.468 ],
       [ 0.9058,  1.1184, -1.0516]])

In []: arr.sort(1)
	   arr
Out[]: 
array([[-1.6331, -0.7139, -0.4959],
       [-1.3132, -0.1935,  0.8236],
       [-1.6748, -0.863 ,  3.0336],
       [-2.468 , -0.3161,  0.5362],
       [-1.0516,  0.9058,  1.1184]])
```

计算数组分位数最简单的办法是对其排序，然后选取特定位置的值：

```python
In []: large_arr = randn(1000)
	   large_arr.sort()
       large_arr[int(0.05 * len(large_arr))]	 # 5% 分位数

Out[]: -1.5791023260896004
```

### 4.唯一值和其他集合逻辑
函数|说明
---|---
unique(x)|计算 x 中的唯一元素，并返回有序结果
intersect1d(x, y)|计算 x 和 y 中的公共元素，并返回有序结果
union1d(x, y)|计算 x 和 y 的并集，并返回有序结果
in1d(x, y)|	得到一个「x 元素是否包含于 y」的布尔型数组
setdiff1d(x, y)|集合的差，即元素在 x 中且不在 y 中
setxor1d(x, y)|集合的对称差，即存在于一个数组中但不同时存在于两个数组中的元素

其中最常用的可能是 np.unique 了，它用于找出数组中的唯一值并返回已排序结果：

```python
In []: ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
	   np.unique(ints)
Out[]: array([1, 2, 3, 4])
```
### 5.生成随机数
如果需要产生大量样本值，np.random 模块比 Python 内置的 random 速度快得多。

函数|说明
---|---
seed |确定随机数生成器的种子
permutation |返回一个序列的随机排列或范围
shuffle |对序列随机排序
rand |产生均匀分布的样本值
randint |在给定的范围内随机选取整数
randn |	产生正态分布（均值为0，标准差为1）的样本值
binomial|产生二项分布样本值
normal|产生正态（高斯）分布样本值
beta|产生 beta 分布样本值
chisquare|产生卡方分布样本值
gamma|产生 gamma 分布样本值
uniform|产生在[0,1) 中均匀分布的样本值

### 参考资料
- [NumPy Basics: Arrays and Vectorized Computation - Python for Data Analysis [Book]](https://www.safaribooksonline.com/library/view/python-for-data/9781449323592/ch04.html)
- [Numerical Python: A Practical Techniques Approach for Industry](https://www.amazon.com/Numerical-Python-Practical-Techniques-Approach/dp/1484205545)