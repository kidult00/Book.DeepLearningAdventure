# DeepLearning-2: 语言模型和 N-gram

语言模式是自然语言处理的一个基础概念。我们可以从语料中得到「语言模型」—— 即句子的概率，可用于：

- 发现错别句子
- 发现新短语
- 生成句子（如[模仿汪峰写歌](https://github.com/phunterlau/wangfeng-rnn)）

机器怎样理解自然语言呢？有两种思路：

- 学习语法：词性、句子成分，但不能保证语义，如，火星追杀绿色的梦
- 概率统计：[齐夫定律](https://www.wikiwand.com/zh-cn/%E9%BD%8A%E5%A4%AB%E5%AE%9A%E5%BE%8B)（词频 $$\propto \frac{1}{rank}$$ ：频率最高的单词出现的频率大约是出现频率第二位的单词的2倍，而出现频率第二位的单词则是出现频率第四位的单词的2倍），香农的信息论

### 概率论基本原理
概率空间：所有可能的结果。概率中的原子结构是基本事件，不可分割，不重叠；分子结构是事件（基本事件的集合）。事件的概率，可以理解为所选取的基本事件在整个空间里占的面积比例。

- 联合概率 P(A,B)：两个事件同时发生，比如掷两次筛子，可能有 $6^2$ 种结果。
- 条件概率 P(B|A)：A 条件下 B 发生的概率。从一个大的空间进入到一个子空间（切片），计算在子空间中的占比。$$P(B|A) = \frac{P(A,B)}{P(A)}$$

### 概率语言模型

- 计算句子的概率： $$P(S) = P(w_1,w_2,w_3,...,w_n)$$
- 用处：句子错误检查、输入法候选、生成有用的句子等等
- 统计：随着空间膨胀，数据变稀疏，样本有效性降低

对句子做最简化的处理，先考虑只有两个词的句子，根据条件概率公式，它的概率等于第一个词的空间占比，乘以第一个词的概率空间中第二个词的占比：$$P(w_1,w_2) = P(w_2|w_1)*P(w_1)$$

最初级的语言模型（Unigram），可以人为地假设词之间是独立的： $$P(w_2|w_1) \approx P(w_2)$$，于是这个句子的概率约等于两个词的频率相乘： $$P(w_2,w_1) \approx P(w_1)*P(w_2)$$

如果把两个词的句子扩展为三个词：$$P(w_1,w_2,w_3) = p(w_1,w_2)*p(w_3|w_1,w_2) = p(w_1)*p(w_2|w_1)*p(w_3|w_1,w_2)$$

以此类推：

$$P(w_1,w_2,...w_n) = \prod_{i} P(w_i|w_1w_2...w_{i-1})$$

这样做的话，对每个词要考虑它前面的所有词，这在实际中意义不大。可以做些简化吗？

我们可以基于马尔科夫假设来做简化。

> 马尔科夫假设是指，每个词出现的概率只跟它前面的少数几个词有关。比如，二阶马尔科夫假设只考虑前面两个词，相应的语言模型是三元模型。引入了马尔科夫假设的语言模型，也可以叫做马尔科夫模型。
> 
> 马尔可夫链（Markov chain）为狀態空間中经过从一个状态到另一个状态的转换的随机过程。该过程要求具备“无记忆”的性质：下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关。


比如对上面公式做一个 i-k 的简化：

$$P(w_1,w_2,...w_n) \approx \prod_{i} P(w_i|w_{i-k}...w_{i-1})$$

物理意义上说，上面的公式意味着每次看到 i 时，只要关注 i 前面的 k 个词，这就是 N-gram 模型的思路。

### 作业
作业 1：$$P(w_1,w_2) = P(w_2|w_1)*P(w_1)$$ 没有减少参数个数，为什么？

作业 2：在自己选取的数据集合上建立 Bigram 模型，并使用该建立好的模型生成句子。

### 其他
技巧：进入 docker 容器的 shell 环境

``docker exec -it container_id /bin/bash``



### Ref

- [齐夫定律](https://www.wikiwand.com/zh-cn/%E9%BD%8A%E5%A4%AB%E5%AE%9A%E5%BE%8B)
- [蒙特卡罗方法入门 - 阮一峰的网络日志](http://www.ruanyifeng.com/blog/2015/07/monte-carlo-method.html)
- [Language Modeling - Course notes for NLP by Michael Collins, Columbia University](http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf)
- [Language Modeling with Ngrams](https://web.stanford.edu/~jurafsky/slp3/4.pdf)
- [4 - 1 - Introduction to N-grams- Stanford NLP - Professor Dan Jurafsky & Chris Manning - YouTube](https://www.youtube.com/watch?v=s3kKlUBa3b0)
- [马尔可夫链 - Wikiwand](https://www.wikiwand.com/zh/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)
- [sunoonlee 同学的笔记](https://github.com/sunoonlee/DeepLearning101/issues/2)
- [DeepLearning101/zhatrix/DeepLearning101](https://github.com/zhatrix/DeepLearning101/blob/master/ch1/project/assignmentch1.ipynb)
