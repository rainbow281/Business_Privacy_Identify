# 2020CCF-NER
2020 CCF大数据与计算智能大赛-非结构化商业文本信息中隐私信息识别-第7名方案

bert base + crf + flat + fgm + swa + pu learning策略 + clue数据集 = test1单模0.906

#FLAT：词汇增强方式,解决计算效率低下、引入词汇信息有损问题,让Transformer适用NER任务
#(https://zhuanlan.zhihu.com/p/235361403)

#Fast Gradient Method (FGM)：对embedding层在梯度方向添加扰动，动机：采用对抗训练缓解模型鲁棒性差的问题，提升模型泛化能力

#swa随机加权平均：改进了基于随机梯度下降(SGD)的深度学习中的泛化，并且可以作为PyTorch中任何其他优化器的替代
#（https://zhuanlan.zhihu.com/p/137083086）

#PU Learning（Positive-unlabeled learning）是半监督学习的一个研究方向，指在只有正类和无标记数据的情况下，训练二分类器
#（https://zhuanlan.zhihu.com/p/82556263）

#clue数据集
词向量：https://github.com/Embedding/Chinese-Word-Vectors SGNS(Mixed-large 综合)

loss mask相关代码为pu learning策略的实现

主要模块版本
python 3.6.9

torch 1.1.0

transformers 3.0.2 

pytorchcrf 1.2.0 

torchcontrib 0.0.2
