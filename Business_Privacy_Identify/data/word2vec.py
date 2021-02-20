import pickle
from global_config import *

ROOT_LOCAL_DATA = ROOT_DATA + 'w2v\\'

# 从sgns.merge.word中提取出词汇和向量并分别写入w2v_vocab.pkl和w2v_vector.pkl
def conv2pkl():
    # raw_file：原始文、 vocab_file：词汇文件、 vec_file：向量文件
    raw_file = ROOT_LOCAL_DATA + 'sgns.merge.word'
    vocab_file = ROOT_LOCAL_DATA + 'w2v_vocab.pkl'
    vec_file = ROOT_LOCAL_DATA + 'w2v_vector.pkl'

    raw_fp = open(raw_file, 'r', encoding='utf-8')
    # 'w+' 是文本写入，'wb+'是字节写入。'w2v_vocab.pkl'和'w2v_vector.pkl'为写入文件，没有则自动创建
    vocab_fp = open(vocab_file, 'wb+')
    vec_fp = open(vec_file, 'wb+')

    # readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表
    # 将原始文件的所有数据（从第二行开始）放入data
    raw_data = raw_fp.readlines()[1:]
    vocab_list = {'PAD': 0}
    # vec_list=[[0.0,0.0,0.0,0.0......]]内部300个0.0
    vec_list = [[0.0] * 300]

    for index, d in enumerate(raw_data):
        # split() 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等，返回分割后的字符串列表。
        d = d.split()
        # 因为索引从0开始，从1开始取所以需要+1
        # 词汇字典中 键：原始文件（除第一行）每行第一个值，值：原始文件的行索引
        vocab_list[d[0]] = index + 1
        # 向量列表中将原始文件（除第一行）第一个值以外的其他值构成的向量列表作为元素
        vec = [float(s) for s in d[1:]]
        vec_list.append(vec)

    # pickle.dump：序列化对象，将对象保存到文件file中去。默认以文本的形式进行序列化
    pickle.dump(vocab_list, vocab_fp)
    pickle.dump(vec_list, vec_fp)


def get_w2v_vocab():
    return pickle.load(open(ROOT_LOCAL_DATA + 'w2v_vocab.pkl', 'rb'))


def get_w2v_vector():
    return pickle.load(open(ROOT_LOCAL_DATA + 'w2v_vector.pkl', 'rb'))


if __name__ == '__main__':
    conv2pkl()


