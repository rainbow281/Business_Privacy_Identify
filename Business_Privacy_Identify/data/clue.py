from global_config import ROOT_DATA
from util.tool import *

ROOT_LOCAL_DATA = ROOT_DATA + 'clue\\'


# ================================================================================================
# 将输入文件中同一整体文本全部以一条{'id': index, 'text': d['text'], 'entities': []}形式存入输出文件
def conv2json_(infiles, outfile, cnum=1):
    # 将输入文件的内容放入data
    # cnum：同一段文本被切分的份数，默认为1
    # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    # append（）在列表末尾添加新的对象
    data = []
    if type(infiles) is list:
        for infile in infiles:
            data.extend(json_load_by_line(infile))
    else:
        data.extend(json_load_by_line(infiles))

    # 将输入的每条数据以每条{'id': index, 'text': d['text'], 'entities': []}形式存入outdata
    outdata = []
    index = 0
    for d in data:
        if d['text'] in ('《蝙蝠侠》', '星际争霸2', '星际2', '反恐精英', '穿越火线', '魔兽争霸3', '《超人》', '《变形金刚》', '英雄联盟'):
            # continue 语句跳出本次循环，而break跳出整个循环
            continue
        # entity实体
        sample = {'id': index, 'text': d['text'], 'entities': []}
        index += 1
        # 判断 键‘label'是否在字典d里，只有dev.json和train.json中存在label键
        if 'label' in d:
            # 三层嵌套循环，获取隐私信息、隐私类型、开始位置、结束位置（闭区间）
            for category in d['label'].keys():
                for privacy in d['label'][category].keys():
                    for entity in d['label'][category][privacy]:
                        sample['entities'].append({'privacy': privacy,
                                                   'category': category,
                                                   'pos_b': entity[0],
                                                   'pos_e': entity[1]})
        outdata.append(sample)

    # 将切分出来的来自同一段文本的多个字典表示组合为一个字典去表示，此处若为1，处理后结果不变
    # 从outdata变成outdata2
    outdata2 = []
    base = 0
    sample = None
    # index为索引0,1,2...,d为枚举对象的元素，d为字典
    for index, d in enumerate(outdata):
        # 这里是同一个文本的第一段的添加
        if (index % cnum) == 0:
            if sample is not None:
                outdata2.append(sample)
            # d['entities'][:]指取d['entities']的全部
            sample = {'id': int(index / cnum), 'text': d['text'], 'entities': d['entities'][:]}
            base = len(d['text'])
        # 以下是同一文本其它段的添加
        else:
            sample['text'] += d['text']
            for e in d['entities']:
                sample['entities'].append({'privacy': e['privacy'],
                                           'category': e['category'],
                                           'pos_b': e['pos_b'] + base,
                                           'pos_e': e['pos_e'] + base})
            base += len(d['text'])
    # 将outdata2写入train_dev_test_data.json文件
    json_dump_by_line(outdata2, outfile)

def conv2json():
    conv2json_([ROOT_LOCAL_DATA + 'train.json',
                ROOT_LOCAL_DATA + 'dev.json',
                ROOT_LOCAL_DATA + 'test.json'],
               ROOT_LOCAL_DATA + 'train_dev_test_data.json')


if __name__ == '__main__':
    pass
    # conv2json_(ROOT_LOCAL_DATA + 'train.json', ROOT_LOCAL_DATA + 'train_data.json')
    # conv2json_(ROOT_LOCAL_DATA + 'dev.json', ROOT_LOCAL_DATA + 'dev_data.json')
    # conv2json_(ROOT_LOCAL_DATA + 'test.json', ROOT_LOCAL_DATA + 'test_data.json')
    # conv2json_([ROOT_LOCAL_DATA + 'test.json', ROOT_LOCAL_DATA + 'dev.json', ROOT_LOCAL_DATA + 'test.json'],
    #              ROOT_LOCAL_DATA + 'train_dev_test_data.json')
