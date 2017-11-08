# 无法打开 Tensorboard

## 环境

```
macOS 10.12.6

compiler   : GCC 5.4.0 20160609
system     : Linux
release    : 4.9.49-moby
machine    : x86_64
processor  : x86_64
CPU cores  : 2
interpreter: 64bit
docker	   : Version 17.09.0-ce-mac35 (19611)

tensorflow : 1.4.0
python	   : jupyter notebook  python2

```

## 现象

打开课程仓库 pull 下来 ch3 中 lecture3.ipynb 文件（课程视频中童老师演示的文件），运行到最后一部分 ``write data``，写入文件**成功**后，直接在 jupyter Notebook 中打开 tensorboard：

![](http://7xjpra.com1.z0.glb.clouddn.com/tensorboard-issue1.png)

在 chrome 和 safari 都打不开 http://f7178713446b:6006 ，chrome 提示

```
This site can’t be reached

f7178713446b’s server DNS address could not be found.
Try running Network Diagnostics.
DNS_PROBE_FINISHED_NXDOMAIN
```

## 尝试 1：定义host
搜索 f7178713446b 的含义，折腾很久发现是 docker image 的编号……

于是参考资料，把命令改成 

``!tensorboard --logdir="tf" --port 6006 --host=127.0.0.1``

运行后提示

```
TensorBoard 0.4.0rc2 at http://127.0.0.1:6006 (Press CTRL+C to quit)
```

依然打不开 http://127.0.0.1:6006 ，chrome 提示

```
This site can’t be reached

127.0.0.1 refused to connect.
Try:
Checking the connection
Checking the proxy and the firewall
ERR_CONNECTION_REFUSED
```

## 尝试 2：另外安装 tensorboard，用 terminal 打开

在 Jupyter Notebook 新建 Terminal ，用 pip 安装 tensorboard：

```bash
pip show tensorboard

Name: tensorboard
Version: 1.0.0a6
Summary: Standalone TensorBoard for visualizing in deep learning
Home-page: https://github.com/dmlc/tensorboard
Author: zihaolucky
Author-email: zihaolucky@gmail.com
License: Apache 2.0
Location: /usr/local/lib/python2.7/dist-packages
Requires: mock, Pillow, numpy, protobuf, wheel, six, werkzeug
```
然后在 Jupyter Notebook 的 terminal 中运行

```
tensorboard --logdir="/Users/kidult/WorkStation/DeepLearning101-002/ch3/tf/" --host=127.0.0.1
```

依然打不开

## 尝试 3：用 python3 环境运行

Jupyter Notebook 的 Python3 下，TensorFlow 版本没有更新，是 1.3.0


运行全部 cell，再次写入 log，运行 ``!tensorboard --logdir="tf" --port 6006 --host=127.0.0.1``

```
Starting TensorBoard b'55' at http://127.0.0.1:6006
(Press CTRL+C to quit)
WARNING:tensorflow:Found more than one graph event per run, or there was a metagraph containing a graph_def, as well as one or more graph events.  Overwriting the graph with the newest event.
```

依然打不开

## 尝试 4：添加端口

根据 lecture3.ipynb 最后几行的提示

>如果在 Docker 环境中进行可视化，可以多暴露一个端口，如 docker run -it -p 1234:1234 -p 8888:8888
>
如果想在已有环境的基础上添加端口，可以参考 ch1 答疑中 Docker 环境使用的 commit + run大法。

>在导出的目录处运行 tensorboard --port 1234 --logdir=tf，本机即可访问 Tensorboard 服务了。

但是没有找到 ``ch1 答疑中 Docker 环境使用的 commit + run大法`` 在哪里……

正在绝望时，在 issue 上看到教练回复小鹤的问题 [24h[QUESTION][CH3]tensorboard, 报错:AttributeError: 'module' object has no attribute 'summary' · Issue #74](https://github.com/AIHackers/DeepLearning101-002/issues/74)

>你需要映射6006端口才可以，你重新运行docker run命令，不过这次多映射一个端口。
docker run -it -p 8888:8888 -p 6006:6006 -v /DL10 2:/notebooks/DL102 --name dlboard102 tensorflow/tensorflow
这种方法比较简单，不过有些package就要重新安装了。

```
docker run -it -p 8888:8888 -p 6006:6006 -v /DL10 2:/notebooks/DL102 --name dlboard102 tensorflow/tensorflow
```

增加了端口映射后，重新打开 Notebook，TensorFlow 版本回到了 1.3.0，运行 ``!tensorboard --logdir="tf" --port 6006 --host=127.0.0.1``

```
Starting TensorBoard 55 at http://127.0.0.1:6006
(Press CTRL+C to quit)
```
依然打不开，浏览器提示变成：

```
This page isn’t working

127.0.0.1 didn’t send any data.
ERR_EMPTY_RESPONSE
```

再尝试去掉 host，运行 ``!tensorboard --logdir="tf" --port 6006``

然后在浏览器打开 http://127.0.0.1:6006/，终于打开了 tensorboard！

T_____T

但！是！奇怪的是，试着用回没有多开端口的 image，也可以打开了！

所以目前的总结是：**运行 ``!tensorboard --logdir="tf" --port 6006``,如果看到用 image id 命名的地址，直接用 127.0.0.1:6006 打开即可** ……


[docker commit | Docker Documentation](https://docs.docker.com/engine/reference/commandline/commit/)