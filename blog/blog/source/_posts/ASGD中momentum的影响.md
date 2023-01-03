---
title: 多节点异步更新中momentum的影响
date: 2017-06-21 12:31:08
category: deep learning
tags: [caffe, deep learning, momentum]
---


这几天的主要工作是将caffe移植到组内新开发的某个计算框架，在验证正确性时遇到一个问题。由于计算框架只支持异步更新的方式，因此采用全异步SGD算法训练Alexnet时非常容易发散。另外调研了一下近期发布的异步更新算法DC-ASGD，实验结果只能说对收敛有些正向效果，仍无法解决训练发散的问题。在另外一个DNN的网络上发现在多机时momentum对收敛结果有较大影响，momentum会导致收敛出现较大波动。

<!-- more -->

网上找了一圈，似乎也就这个有些参考价值：
http://stanford.edu/~imit/tuneyourmomentum/theory/

看来近期得做一些调momentum和学习率的实验了。。。
