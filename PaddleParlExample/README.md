# PaddleParl盘模心得

使用PaddleParl的框架能够大幅降低建模的复杂度。使得算法人员能够更加专注于问题本身，设计更好的模型构建方案。

对应不同场景时，应考虑到模型容量与问题规模之间的匹配度。可以对模型神经网络规模进行以量级为单位的初步搜索。

学习率需要在不同阶段进行设计，较大的学习率能够在模型更新的初期快速找到方向，但会让模型后期无法收敛到一个好的结果。过小的学习率，在模型训练的后期，有利于模型的收敛，但全程使用小学习率训练会导致收敛速度过慢，训练的过程中也很有肯能陷入一个GAP无法收敛。

对于一些特定的问题，可以将一些先验信息引入模型结构，方便模型收敛。

使用不同算法的时候，第一步就是要了解超参数的意义，这样才能更准确的进行调参。