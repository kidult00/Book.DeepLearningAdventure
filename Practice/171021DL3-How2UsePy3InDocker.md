# 如何在 Docker 里切换 Python 版本

​在 DeepLearning-1：Docker 入门和用 Python 实现词频统计 一文中，我们介绍了 Docker 环境的安装和使用。

然后，00 遇到了 Python 的经典问题：Python 2 还是 Python 3？TensorFlow image 默认安装的是 Python2，如果想在 Jupyter Notebook 里使用 Python3，怎么办呢？

在 [TensorFlow 的 这个 Issue](https://github.com/tensorflow/tensorflow/issues/10179) 可以看到，2017年5月已经支持[用 tag 提供不同的 image](https://hub.docker.com/r/tensorflow/tensorflow/tags/)。比如 ``tensorflow/tensorflow:latest-py3`` 就可以（安装并）打开 Python3 环境。

结合目录映射的需要，输入命令完成映射并在 python3 环境下打开：

```bash
docker run -it -p 8888:8888 -v ~/WorkStation/DeepLearning101-002/:/WorkStation/DeepLearning101-002 -w /WorkStation/DeepLearning101-002 tensorflow/tensorflow:latest-py3
```

然后用``docker ps -a``查看所有 image，然后使用命令 ``docker rename CONTAINER ID XXX``，将默认的 Python2 的 image 重命名为 dl，将 Python3 的 image 重命名为 dlpy3：

```bash
CONTAINER ID        IMAGE                              COMMAND                  CREATED             STATUS                      PORTS               NAMES
f46533729239        tensorflow/tensorflow:latest-py3   "/run_jupyter.sh -..."   11 minutes ago      Exited (0) 6 minutes ago                        dlpy3
f7178713446b        tensorflow/tensorflow              "/run_jupyter.sh -..."   42 minutes ago      Exited (0) 15 minutes ago                       dl
```
以后就可以根据需要，打开不同 Python 环境的 image。

``docker start -i dl`` 打开 Python2 环境：

![](http://7xjpra.com1.z0.glb.clouddn.com/docker_py2.png)

``docker start -i dlpy3`` 打开 Python3 环境：

![](http://7xjpra.com1.z0.glb.clouddn.com/docker_py3.png)


参考

- [Docker Image with Python 3? · Issue #3467 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/3467)
- [Support python3 on Docker image tensorflow/tensorflow:latest · Issue #10179 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/10179)