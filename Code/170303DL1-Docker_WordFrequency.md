# DeepLearning-1：Docker 入门和用 Python 实现词频统计


## 一、神经网络简介

神经网络简史：

- 40年代：概念雏形（没有学习算法）
- 50年代：可用的学习算法 - 感知机
- 1969年：Minsky 泼冷水
- 70年代：BP 算法，训练多层神经网络
- 90年代：SVM 支持向量机「打败」神经网络
- 2006：深层网络理论、实验上有所突破
- 2012：ImageNet，大幅提升结果（错误率 15.3%）

人工神经网络简单来说，就是在输入层和输出层中间加入多个隐层，实现多层神经元信号处理。它是一种从底层构建的思路。

- Top-down approach to AI ：西蒙为代表的符号学派
	- 形式化方法，将知识表示为符号
	- 运用逻辑进行推理
	- 对自然语言、图像问题基本毫无办法
- Bottom-up approach to AI：神经网络，从最底层（神经元）开始构建



深度学习框架很多，不需要纠结使用哪种框架：

- Tensorflow：Google 开源主推，是最流行的框架，文档齐全。底层是 C++ ，如果对性能要求不高，用 Python 开发的效率更高
- MXNet：亚马逊主推
- Caffe：图形领域，自然语言处理稍弱
- Torch：Facebook 主推


## 二、Docker 环境安装和配置

Docker 是什么？它是一种容器化技术的实现，可以理解为一个轻量级的虚拟环境。

之前 00 被 Python 的版本和各种包虐过，所以折腾了 [Virtualenv](https://virtualenv.pypa.io/en/stable/) 的方法，一个项目新建一个 Python 环境。那么 Docker 跟 Virtualenv 的区别是什么呢？

> Docker completely isolates the TensorFlow installation from pre-existing packages on your machine. The Docker container contains TensorFlow and all its dependencies. 

Docker 有一个 Image 的概念，可以理解为别人已经制作好的环境（类似安卓手机装机软件），把 Python + TensorFlow + Jupyter Notebook 打包好。

### Docker 安装和配置步骤

第一步：下载 [Docker Community Edition for Mac](https://store.docker.com/editions/community/docker-ce-desktop-mac?tab=description)安装。

用 ``docker version`` 可以查看版本：

```
kidults-NMB:~ kidult$ docker version
Client:
 Version:      17.03.0-ce
 API version:  1.26
 Go version:   go1.7.5
 Git commit:   60ccb22
 Built:        Thu Feb 23 10:40:59 2017
 OS/Arch:      darwin/amd64

Server:
 Version:      17.03.0-ce
 API version:  1.26 (minimum version 1.12)
 Go version:   go1.7.5
 Git commit:   3a232c8
 Built:        Tue Feb 28 07:52:04 2017
 OS/Arch:      linux/amd64
 Experimental: true
```

使用 Docker 时，命令行相当于客户端，服务器端在安装完成后需要启动。

第二步：到 [Docker hub](https://hub.docker.com/) 找到 TensorFlow Image ，以此为模板构建自己的容器。参考 [Using TensorFlow via Docker](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/README.md)，用 ``run`` 命令加载 TensorFlow image：

``docker run -it -p 8888:8888 tensorflow/tensorflow``

开始下载后，发现速度非常慢：

![](http://7xjpra.com1.z0.glb.clouddn.com/download_tensorflow_image.png)

参考小伙伴的 [Docker Hub Mirror加速Docker官方镜像下载](http://www.jianshu.com/p/d896ec46db66) 笔记，使用镜像下载。在 Mac 上配置[加速器](https://www.daocloud.io/mirror#accelerator-doc)很简单，右键点击桌面顶栏的 docker 图标，选择 Preferences ，在 Advanced 标签下的 Registry mirrors 列表中加入镜像地址: ``http://d43d99f5.m.daocloud.io Copy``，点击 Apply & Restart 按钮使设置生效。

![](http://7xjpra.com1.z0.glb.clouddn.com/docker_mirror.png)

再次运行``docker run -it -p 8888:8888 tensorflow/tensorflow``，速度飞了起来~

下载成功后，可以在浏览器看到 Jupyter Notebook 界面：

![](http://7xjpra.com1.z0.glb.clouddn.com/docker-image-loaded.png)

### Docker 常用命令

- 关闭 docker：control+c
- 查看运行状态：``docker ps`` 
- 把本地目录映射到容器：``docker run -it -p 8888:8888 -v 原路径:目标路径 tensorflow/tensorflow``
- 查看历史容器：``docker ps -a`` 

	```
	kidults-NMB:my_venv kidult$ docker ps -a
	CONTAINER ID        IMAGE                   COMMAND             CREATED             STATUS                      PORTS                              NAMES
	f4eaa7828ac5        tensorflow/tensorflow   "/run_jupyter.sh"   7 minutes ago       Up About a minute           6006/tcp, 0.0.0.0:8888->8888/tcp   priceless_mccarthy
	b09563e0479b        tensorflow/tensorflow   "/run_jupyter.sh"   52 minutes ago      Exited (0) 7 minutes ago                                       silly_chandrasekhar
	d669d764de48        tensorflow/tensorflow   "/run_jupyter.sh"   About an hour ago   Exited (0) 52 minutes ago                                      relaxed_davinci
	```
- 恢复历史容器：``docker start -i id`` (ID只需要写前几位)
- 删除容器：``docker rm id`` 
- 重命名容器：``docker rename CONTAINER ID XXX``(重命名为 dl)
- 在容器里安装包：``!pip install xxx``


参考 [junjielizero 同学的笔记](https://github.com/junjielizero/DeepLearning101/blob/master/ch0/note/README.md)，优化了几个步骤：

1. 创建跟 github 同步的容器：

	```
	docker run -it -p 8888:8888 -v ~/Workspace/DeepLearning101/:/Workspace/DeepLearning101 -w /Workspace/DeepLearning101 tensorflow/tensorflow 
	```
2. 给 docker 启动命令起别名

	把课程容器改名为``dl``，然后新建别名 ``alias dsdl="docker start -i dl"``

3. 在 iTerm2 运行 ``dsdl``，按着 Command 点击链接 localhost:8888 即可打开项目

## 三、词频统计作业

> 统计 [happiness.txt](https://github.com/AIMinder/DeepLearning101/blob/master/ch0/code/happiness.txt) 的词频

思路：读取文件 → 用 jieba 分词 → 清除非中文字符 → 用 counter 计数 → 用 sorted 排序

1. 读取文件

	```python
	loadfile = open('happiness.txt', 'r')
	text = loadfile.read().decode('utf-8')
	loadfile.close()
	```

2. 用 jieba 分词

	```python
	words = jieba.cut(text)
	```

3. 清除非中文字符
	
	> re.match(pattern, string, flags=0)
	
	>If zero or more characters at the beginning of string match the regular expression pattern, return a corresponding MatchObject instance. Return None if the string does not match the pattern.
	
	```python
	for word in words:
    	if re.match(u'([\u4e00-\u9fff]+)', word):
   		     segments.append(word)
	```


4. 用 Counter 计数，用 sorted 排序

	Counter 用法
	
	> collections.Counter([iterable-or-mapping])
	
	> A Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values.
	
	sorted 用法
	
	> sorted(iterable[, cmp[, key[, reverse]]])
	Return a new sorted list from the items in iterable.

	> cmp specifies a custom comparison function of two arguments which should return a negative, zero or positive number depending on whether the first argument is considered smaller than, equal to, or larger than the second argument: cmp=lambda x,y: cmp(x.lower(), y.lower()). The default value is None.

	lambda 表达式用法
	
	> 通常在需要一个函数但是又不想命名一个函数时使用，即匿名函数。比如实现一个可以求list中所有元素和的函数：
	
	>```
	from functools import reduce 
	l = [1,2,3,5,-9,0,45,-99] 
	reduce(lambda x,y:x+y,l)
	```

	在这里，用 Counter 统计字典中词的出现次数，以 lambda 取字典中的 value 值（key 是 x[0]，value 就是 x[1]）用 sorted 方法按降序排序：

	```python
	sorted_list = sorted(Counter(dict).items(), key=lambda x:x[1], reverse=True)
	```

最后，完整的代码和结果如下：

```python
# -*- coding: utf-8 -*-

import jieba # for spliting
import re # for regular expression
from collections import Counter # for stat

# Read file
loadfile = open('happiness.txt', 'r')
text = loadfile.read().decode('utf-8')
loadfile.close()

# Split words into a dict
dict = []
words = jieba.cut(text)
for word in words:
    if re.match(u'([\u4e00-\u9fff]+)', word):
        dict.append(word)

# Sort the list
sorted_list = sorted(Counter(dict).items(), key=lambda x:x[1], reverse=True)

# Print result
for i in sorted_list[:10]:
    print " '%s' : %d " % (i[0], i[1])
```
结果：

     '的' : 22848 
     '是' : 4123 
     '在' : 3538 
     '他' : 2522 
     '了' : 2288 
     '人' : 2089 
     '他们' : 1811 
     '和' : 1746 
     '有' : 1478 
     '我' : 1433 



### Ref

- [Python开发生态环境简介](https://github.com/dccrazyboy/pyeco/blob/master/pyeco.rst)
- [Installing TensorFlow on Mac OS X ](https://www.tensorflow.org/install/install_mac)
- [tensorflow/tensorflow - Docker Hub](https://hub.docker.com/r/tensorflow/tensorflow/)
- [Using TensorFlow via Docker](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/README.md)
- [Docker学习笔记(2)--使用Docker Hub Mirror加速Docker官方镜像下载](http://www.jianshu.com/p/d896ec46db66)
- [配置 Docker 加速器](https://www.daocloud.io/mirror#accelerator-doc)
- [Speed Up Your Terminal Workflow with Command Aliases and .profile](https://computers.tutsplus.com/tutorials/speed-up-your-terminal-workflow-with-command-aliases-and-profile--mac-30515)
- [fxsjy/jieba: 结巴中文分词](https://github.com/fxsjy/jieba)
- [Lambda 表达式有何用处？如何使用？](https://www.zhihu.com/question/20125256)