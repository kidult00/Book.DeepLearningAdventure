# 菜鸟数据科学入门05 - 随机漫步练习

![](http://7xjpra.com1.z0.glb.clouddn.com/170108Random_Walk_example.svg.png)

> 随机漫步（Random Walk）是一种数学统计模型，由一连串的轨迹所组成，其中每一次都是随机的。它能用来表示不规则的变动形式，如同一个人酒后乱步，所形成的随机过程记录。1905年由卡尔·皮尔逊首次提出。——Wiki

本练习通过模拟随机漫步来说明如何运用 NumPy 进行数组运算。

### 用 Python 实现

``` python
# -*- coding: utf-8 -*-

# 1.需要一个表示漫步位置的变量 position
# 2.需要一个存放位置序列的数组 walk
# 3.需要一个控制数组个数的变量 step
# 5.对 step 个 position，随机让 position 向上或向下走
# 6.把 position 存放到数组 walk 里

import random
position = 0
walk=[]
step = 100

for i in xrange(step):
    foot = 1 if random.randint(0,1) else -1  # randint 生成 0~1 的随机整数，即要么 0 要么1
    position += foot
    walk.append(position)

print walk
```
结果：

```
[1, 0, 1, 2, 1, 0, -1, -2, -3, -2, -3, -4, -5, -4, -5, -6, -5, -4, -3, -4, -5, -6, -5, -6, -5, -4, -3, -2, -1, 0, -1, -2, -3, -2, -3, -2, -1, 0, -1, 0, -1, -2, -1, -2, -3, -4, -3, -4, -5, -4, -5, -6, -7, -8, -9, -10, -9, -8, -9, -10, -11, -12, -11, -12, -11, -12, -11, -10, -11, -12, -13, -14, -13, -12, -13, -12, -13, -12, -11, -10, -11, -10, -9, -8, -9, -8, -9, -8, -9, -10, -9, -10, -9, -8, -9, -8, -7, -6, -5, -6]
```
### 用 NumPy 数组实现

```python
# -*- coding: utf-8 -*-

import numpy as np
%matplotlib inline

nsteps = 100
draws = np.random.randint(0,2, size = nsteps) # 随机生成 0~1 的 nsteps 个整数
steps = np.where(draws > 0, 1, -1) # where(condition, true_result, false_result)
walk = steps.cumsum() # 计算所有 steps 累计和

print 'Random walk:' 
print steps
print 'Map:'
print walk
print 'Max in the array:', walk.min()
print 'Min in the array:', walk.max()
plot(walk);
```

输出：

```
Random walk:
[-1 -1  1 -1  1 -1  1 -1  1  1  1 -1 -1  1  1  1 -1 -1  1 -1  1  1  1 -1  1
 -1 -1  1 -1  1  1  1  1  1 -1 -1  1 -1  1  1 -1 -1  1  1  1  1 -1 -1 -1 -1
 -1 -1 -1  1 -1 -1  1 -1 -1  1 -1  1  1 -1 -1  1  1 -1  1  1  1 -1  1  1 -1
  1  1  1  1 -1  1 -1 -1 -1 -1 -1  1 -1  1  1  1 -1 -1 -1 -1 -1  1  1  1 -1]
Map:
[-1 -2 -1 -2 -1 -2 -1 -2 -1  0  1  0 -1  0  1  2  1  0  1  0  1  2  3  2  3
  2  1  2  1  2  3  4  5  6  5  4  5  4  5  6  5  4  5  6  7  8  7  6  5  4
  3  2  1  2  1  0  1  0 -1  0 -1  0  1  0 -1  0  1  0  1  2  3  2  3  4  3
  4  5  6  7  6  7  6  5  4  3  2  3  2  3  4  5  4  3  2  1  0  1  2  3  2]
Max in the array: -2
Min in the array: 8
```
![](http://7xjpra.com1.z0.glb.clouddn.com/wPIasx201fcoAAAAABJRU5ErkJggg==.png)

生成二维的随机漫步：

```python
# -*- coding: utf-8 -*-
# two dimension

nsteps = 100
draws = np.random.randint(-1,2,size=(2,nsteps)) 
# 生成 2 行 100 列的数组，从 (-1,0,1) 中随机挑选

walk = draws.cumsum(1) # 沿 y 轴对每一行求和
print(walk)
plot(walk[0,:],walk[1,:]);
```
输出：

```
[[  1   1   2   3   2   3   3   2   3   2   3   3   3   3   4   4   3   2
    1   0   0   0  -1  -2  -3  -4  -3  -3  -2  -1  -2  -2  -3  -4  -3  -4
   -4  -4  -5  -6  -7  -8  -7  -8  -9 -10 -10 -11 -12 -13 -14 -14 -15 -15
  -15 -16 -15 -15 -15 -16 -17 -16 -17 -18 -17 -18 -18 -18 -18 -19 -20 -19
  -19 -19 -18 -17 -16 -16 -17 -16 -17 -18 -19 -19 -20 -20 -21 -21 -20 -20
  -20 -19 -18 -19 -18 -19 -20 -19 -19 -19]
 [  1   0   0   1   2   2   3   4   3   4   3   4   3   2   1   1   2   2
    3   2   3   3   2   3   4   3   3   3   2   3   2   3   3   3   2   2
    1   1   2   1   1   1   1   2   1   1   1   1   0  -1   0  -1  -1  -2
   -1  -1  -2  -1   0  -1  -1   0   0  -1  -2  -3  -3  -2  -1  -2  -1  -1
   -1  -1   0   0   1   0  -1  -1  -2  -3  -2  -3  -3  -3  -3  -2  -2  -2
   -1  -2  -2  -1   0   0   1   0  -1   0]]
```

![](http://7xjpra.com1.z0.glb.clouddn.com/170108Random_Walk_output2.png)

参考：

* [Random walk - Wikipedia](https://en.wikipedia.org/wiki/Random_walk)
* [Numpy 练习题 ](http://www.cnblogs.com/NaughtyBaby/p/5500132.html)
* [pandas-notebook · GitBook](https://www.gitbook.com/book/amaozhao/pandas-notebook/details)