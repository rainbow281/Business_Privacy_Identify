import torch

# import as只是改名字
import data.ccf20202 as ccf2020
import data.clue as clue
import data.word2vec as w2v
# 数据模块导入：数据配置、数据集仓库、数据处理
from data.ccf20202 import DataConfig, CCF2020DataSetFactory, CCF2020DataProcess

# 模块部分导入：
# 模块编码：编码器配置、字编码
from module.encoder import EncoderConfig, WordE
# 模块融合：融合配置、融合编码
from module.fusion import FusionConfig, FusionE
# 模块输出：输出配置、输出配置对象、便签编码
from module.output import OutputConfig, OutputConfigObie, LabelE
# 模块装配：装配配置、命名实体识别模型装配
from module.fitting import FittingConfig, NERModelFitting
# 模块快速原型：快速原型
from model.fast_prototype import FastPrototype

# 全局配置：根权重、根结果、根数据
from global_config import ROOT_WEIGHT, ROOT_RESULT, ROOT_DATA

# 导入标准库os
import os

# 配置类：初始化、字符串输出配置信息
class Config(object):
    def __init__(self):
        # 生成各类配置对象实例：
        # 数据配置对象、编码器配置对象、融合配置对象、输出配置对象、输出配置对象的对象、装配（拟合）配置对象
        self.data = DataConfig()
        self.encoder = EncoderConfig()
        self.fusion = FusionConfig()
        self.output_type = OutputConfig()
        self.output = OutputConfigObie()
        self.fitting = FittingConfig()

        # 设置数据配置对象：
        # 1.数据最大长度=140、2.数据的分词器（roberta-wwm-ext预训练语言模型）、3.数据类型数量=14
        self.data.max_len = 140  # max token length
        self.data.tokenizer = self.encoder.ptm_model
        self.data.num_types = 14

        # 设置编码器配置对象:
        # 字编码类型=word2vec
        # self.encoder.ptm_model = 'hfl/chinese-roberta-wwm-ext-large'
        # self.encoder.ptm_feat_size = 1024
        # self.encoder.num_ptm_layers = 24
        self.encoder.worde = WordE.w2v

        # 设置融合配置对象：
        # 1.融合编码类型=flat、2.en_ffd（未知属性意思）=false
        # 3.位置数=4（true）或1（false）、4.输入特征层大小=编码器配置对象的输出特征层大小
        self.fusion.fusion = FusionE.flat
        self.fusion.en_ffd = False
        self.fusion.num_pos = 4 if self.fusion.fusion == FusionE.flat else 1
        self.fusion.in_feat_size = self.encoder.out_feat_size

        # 设置输出配置对象：
        # 1.类型数目=14、2.输入特征层大小=编码配置对象的输出特征层大小
        self.output_type.num_types = 14
        self.output_type.in_feat_size = self.encoder.out_feat_size

        # 设置输出配置对象的对象：
        # 1.便签编码=Nobie、2.类型数目=14、3.输入特征层大小=融合编码配置对象的输出特征层大小
        self.output.label = LabelE.Nobie
        self.output.num_types = 14
        self.output.in_feat_size = self.fusion.out_feat_size

        # 设置装配配置对象：
        # 1.测试模式=false、2.调整与否=false、3.使通过=true、4.折叠数目=5
        # 5.开发速度=0.2、6.使用fgm与否=true、7.使用swa与否=true、8.时期数=8
        # 9.最后时期数=8、10.批次大小=16、11.学习率字典设置（如果输出配置对象的对象的标签！=便签编码的point，则修改crf的学习率=0.005）
        # 12.是否冗长=true
        self.fitting.test_mode = False
        self.fitting.reshuffle = False  # 记得去改下随机种子
        self.fitting.en_cross = True
        self.fitting.fold_num = 5
        self.fitting.dev_rate = 0.2
        self.fitting.en_fgm = True
        self.fitting.en_swa = True
        self.fitting.epochs = 8
        self.fitting.end_epoch = 8
        self.fitting.batch_size = 16
        self.fitting.lr = {'ptm': 0.00003,
                           'other': 0.00003}
        if self.output.label != LabelE.point:
            self.fitting.lr['crf'] = 0.005
        self.fitting.verbose = True

    # 输出字符串，如果直接打印实例化对象，则并非输出内存地址，而是执行此方法
    def __str__(self):
        string = ""
        string += str(self.encoder)
        string += str(self.fusion)
        string += str(self.output)
        string += str(self.data)
        string += str(self.fitting)
        return string

# =====================================================================================================
if __name__ == '__main__':
    config = Config()
    # 使用python获取系统信息 os.environ（）
    print('\nCUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])
    print(config)
    # 是否进行数据准备
    en_prep = True
    # 是否进行格式化
    en_format = True
    # 是否进行训练
    en_train = True
    # 不启用 BatchNormalization（对网络中间的每层进行归一化处理，保证每层提取的特征分布不会被破坏）
    # 和 Dropout（在每个训练批次中，通过忽略一半的特征检测器，可以明显的减少过拟合现象），保证BN和dropout不发生变化，
    # pytorch框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，
    # 不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果。
    en_eval = True
    # 是否进行训练
    en_test = True
    # 是否进行转换
    en_conv = True

    # 传入数据配置类的对象生成数据处理类实例
    data_process = CCF2020DataProcess(config.data)

    # 进行数据准备工作
    if en_prep:
        clue.conv2json()
        ccf2020.conv2json()
        w2v.conv2pkl()

    # 生成中间文件 seg_xxx_data.json, token_xxx_data.json, map_xxx_data.json
    if en_format:
        data_process.format_data('train', num_works=4)
        data_process.format_data('test', num_works=4)
        # data_process.format_data('train_loss_mask', num_works=4)

    data_factory = CCF2020DataSetFactory(config.fitting)

    model_fitting = NERModelFitting(config, data_process.generate_results)

    # train
    best_weight_list = []
    for fold_index in range(0, config.fitting.fold_num if config.fitting.en_cross else 1):
        if en_train or en_eval or en_test:
            model = FastPrototype(config).cuda()
        else:
            model = None
        # train_data = data_factory({'type_data': 'train_loss_mask', 'fold_index': fold_index})
        train_data = data_factory({'type_data': 'train', 'fold_index': fold_index})
        dev_data = data_factory({'type_data': 'dev', 'fold_index': fold_index})
        test_data = data_factory({'type_data': 'test', 'fold_index': fold_index})

        if en_train:
            inputs = {'model': model,
                      'train_data': train_data,
                      'dev_data': dev_data,
                      'test_data': test_data,
                      'dev_res_file': ROOT_RESULT + 'dev_result.json',
                      'test_res_file': ROOT_RESULT + 'test_result.json',
                      'epoch_start': 0}
            _ = model_fitting.train(inputs)
            if config.fitting.en_cross:
                torch.save(model.state_dict(), ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index))
            else:
                torch.save(model.state_dict(), ROOT_WEIGHT + 'swa_model.ckpt')

        if en_eval:
            inputs = {'model': model, 'data': dev_data, 'type_data': 'dev'}
            if config.fitting.en_cross:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index)
                inputs['outfile'] = ROOT_RESULT + 'dev_result_{}.json'.format(fold_index)
            else:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model.ckpt'
                inputs['outfile'] = ROOT_RESULT + 'dev_result.json'
            print(model_fitting.eval(inputs))

        if config.fitting.en_cross and fold_index == config.fitting.fold_num - 1 or False:
            ccf2020.combine_dev_result(ROOT_RESULT, config.fitting.fold_num,
                                       ROOT_RESULT + 'dev_result_all.json')
            ccf2020.analyze_dev_data(ROOT_RESULT + 'dev_result_all.json',
                                     ccf2020.ROOT_LOCAL_DATA + 'seg_train_data.json',
                                     verbose=True)

        if en_test:
            inputs = {'model': model, 'data': test_data, 'type_data': 'test'}
            if config.fitting.en_cross:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model_{}.ckpt'.format(fold_index)
                inputs['outfile'] = ROOT_RESULT + 'test_result_{}.json'.format(fold_index)
            else:
                inputs['weight'] = ROOT_WEIGHT + 'swa_model.ckpt'
                inputs['outfile'] = ROOT_RESULT + 'test_result.json'
            model_fitting.test(inputs)

    if en_conv:
        if config.fitting.en_cross:
            data_process.combine_by_vote([ROOT_RESULT + 'test_result_{}.json'.format(i) for i in range(config.fitting.fold_num)],
                                         ROOT_RESULT + 'test_result_vote.json')
            ccf2020.mix_clue_result(ROOT_RESULT + 'test_result_vote.json',
                                    ROOT_DATA + 'clue/train_dev_test_data.json',
                                    ROOT_RESULT + 'test_result_vote_mix_clue.json')
            ccf2020.convert2csv(ROOT_RESULT + 'test_result_vote_mix_clue.json',
                                ROOT_RESULT + 'predict_vote.csv')
        else:
            ccf2020.mix_clue_result(ROOT_RESULT + 'test_result.json',
                                    ROOT_DATA + 'clue/train_dev_test_data.json',
                                    ROOT_RESULT + 'test_result_mix_clue.json')
            ccf2020.convert2csv(ROOT_RESULT + 'test_result_mix_clue.json',
                                ROOT_RESULT + 'predict.csv')
