# 菜鸟数据科学入门02 - IPython 基础

![](http://7xjpra.com1.z0.glb.clouddn.com/170918IPython_basic_title.jpg)

## 为什么要用 IPython？

[IPython](http://ipython.org/) (Interactive Python 的简写) 是一个强大的交互式 Python Shell，由 Fernando Perez 在 2001 发起。它的目标是提供 ``Tools for the entire life cycle of research computing.`` 如果说 Python 是数据科学操作的引擎，那么 IPython 就是交互式的控制面板。

IPython 的优点：

- 交互式和批处理功能
- 提高编写、测试速度，执行结果立即可见，方便调试
- 同时保存脚本和计算过程，可重复可互动
- 丰富的数据可视化工具
- 能在本地计算机上对远程服务器中的数据进行分析
- 兼容 markdown 语法，满足数据分析、课程教学、博客写作等需求

Python Scientific Ecosystem

![](http://image.slidesharecdn.com/1idanielrodriguez-160614230356/95/connecting-python-to-the-spark-ecosystem-3-638.jpg?cb=1465945555)

IPython 只是为 NumPy、Scipy、Pandas、Matplotlib 等包提供一个交互式接口，本身并不提供科学计算的功能。这些工具组合在一起，形成了可以匹敌如 Matlab、Mathmatic 等复杂工具的科学计算框架。

	
## Shell or Notebook?
使用 IPython 有两种方式（前提是已经[安装好 IPython](http://jupyter.readthedocs.io/en/latest/install.html)）。

### 在命令行中使用

在命令行中输入 ``ipython``，进入 IPython 环境：

``` 
Python 2.7.10 (default, Jul 30 2016, 19:40:32)
Type "copyright", "credits" or "license" for more information.

IPython 5.1.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]:

```

### 在 Jupyter Notebook 中使用

[Jupyter Notebook](http://jupyter.org/) 是基于浏览器的 IPython shell 图形界面，非常适合用于开发、协同、分享甚至是发布数据科学研究成果。Notebook 以 JSON 格式保存整个交互式会话，可以兼容 Python 代码，文本标记语言如 Markdown，图片，视频，媒体内容等。 IPython Notebook 在 Python 社区中越来越普遍，特别是在科学研究与教育领域，很多课程/博客/书籍都是用 Notebook 写的。

[安装 Jupyter](http://jupyter.org/install.html) 后，在命令行输入 ``jupyter notebook``，进入 IPython 环境

```
[I 15:28:38.305 NotebookApp] Serving notebooks from local directory: /Users/kidult/Workspace/PresentWrok/OpenMind/2016.OM.DS/my_venv
[I 15:28:38.305 NotebookApp] 0 active kernels
[I 15:28:38.305 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/
[I 15:28:38.305 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

浏览器会打开本地的 Jupyter 控制台，新建或选择 .ipynb 后缀文件打开，就可以看到 Notebook 的页面了：

![](http://jupyter.org/assets/jupyterpreview.png)

## Notebook 基本结构和操作

在 Jupyter Notebook 中，基本的操作单元是 cell：

![](http://7xjpra.com1.z0.glb.clouddn.com/IP.SS.cell.jpg)

每个 cell 由 ``In``和``Out``，即输入和输出部分组成。在这里，IPython 实际上生成了名叫 ``In``和``Out`` 的 Python 变量，把操作历史存储起来，以便随时调用。（``_``可调用上一个输出，``_X``可调用 ``Out[X]`` 的输出）

IPython 中的 cell 支持 Markdown 格式，所以很适合用来生成带源码和运行结果的文档。

![](http://7xjpra.com1.z0.glb.clouddn.com/IP.SS.celltype.jpg)

Notebook 中常用操作

命令模式 (按 ``Esc`` 键进入)

键盘|命令|操作
---|---|---
↩ | 回车|进入编辑模式
⇧↩ |shift+回车|运行 cell，选择下一个 cell
M|M 键|切换为 Markdown 格式
A|A 键|在上方插入 cell 
B|B 键|在下方插入 cell 
D,D|按两次 D 键|删除所选 cell
H|H 键|显示快捷键

编辑模式 (按 ``Enter`` 键进入)

键盘|命令|操作
---|---|---
⌘]|command + ]|缩进
⌘[|command + ]|取消缩进
⌘Z|command + Z |撤销
⌘⇧Z|command + shift + Z|重做
⌘↑|command + up 键|跳到 cell 开头
⌘↓|command + down 键|跳到 cell 结尾


## 实用功能

### Tab自动补全
在输入命令时按下 tab 键，可以查看补全选项：

![](http://7xjpra.com1.z0.glb.clouddn.com/IP.SS.tab.jpg)


### 用 ``?`` 查询文档
比如，要查询 ``len`` 的用法，只需要输入 ``len?``

```
In [2]: len?

Type:        builtin_function_or_method
String form: <built-in function len>
Namespace:   Python builtin
Docstring:
len(object) -> integer

Return the number of items of a sequence or mapping.
```

另外，用 ``??`` 可以查看源代码，比如输入``square??``

```
In [8]: square??

Type:        function
String form: <function square at 0x103713cb0>
Definition:  square(a)
Source:
def square(a):
    "Return the square of a"
    return a ** 2
```

### 魔法命令

通过``%lsmagic``即可查看，以下为部分命令：
	
命令|	命令说明
---|---
%hist	|查询输入的历史
%reset	|清空 namespace
%time	|显示 Python语句的执行时间，包括 cpu time 和 wall clock time
%timeit	|显示 Python语句的执行时间，但是这个命令会多次执行相应的语句（可以指定次数）``%timeit``  
%bookmark	|用于存储常用路径
%cd	|进入目录命令
%env	|显示系统环境变量
%pushd dir	|将当前目录入栈，并进入dir指定的目录



### 参考资料

- [Python Data Science Handbook](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb)
- [IPython.org](http://ipython.org/)
- [学习IPython进行交互式编程和数据可视化](https://www.gitbook.com/book/itacey/learning_ipython)
- [使用IPython有哪些好处？ - 知乎](https://www.zhihu.com/question/51467397)
- [为什么要使用IPython？ - 简书](http://www.jianshu.com/p/61f8f7a68bbe)