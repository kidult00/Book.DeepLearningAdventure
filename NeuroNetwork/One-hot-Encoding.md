One-Hot Encoding: transforming labels into one-hot encoded vectors

```python
import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer()

# Here the encoder finds the classes and assigns one-hot vectors
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
lb.transform(labels)
>>> array([[1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0]])
```
![](https://ds055uzetaobb.cloudfront.net/image_optimizer/f4d9e46eb17e36de5fad8796f68908ae5f000ce0.png)


### Cross-Entropy 交叉熵

当我们使用 Softmax，输出是一个向量。可以用 one-hot encoding 将输出表示为向量。

比如说，识别数字的例子中，识别数字 4 的 label vector 是： y = [0,0,0,0,1,0,0,0,0,0]

output prediction vector 可能是：

$\hat y$ = [0.047,0.048,0.061,0.07,0.330,0.062,0.001,0.213,0.013,0.150]

我们希望误差与这些向量的距离成正比。为了计算这个距离，我们将使用交叉熵。

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589b18f5_cross-entropy-diagram/cross-entropy-diagram.png)
