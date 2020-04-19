import os
import sys
import jieba
import numpy as np

'''
1.分词
2.词语转换为ID，统计词频
3.label->id
'''

train_file = 'data/train.txt'
val_file = 'data/val.txt'
test_file = 'data/test.txt'

segment_train_file = 'data/seg/seg_train.txt'
segment_val_file = 'data/seg/seg_val.txt'
segment_test_file = 'data/seg/seg_test.txt'

vocab_file = 'data/vocab_file.txt'
category_file = 'data/category.txt'


def generate_word_seg_file(input_file, output_seg_file):
    '''
    对输入文件分词，并输出到指定文件中。

    Args:
        input_file: 原始数据
        output_seg_file: 分词数据
    '''
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_seg_file, 'w', encoding='utf-8') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '{}\t{}\n'.format(label, word_content)
            f.write(out_line)


def generate_vocab_file(input_seg_file, output_vocab_file):
    '''
    将词转换为id，并统计词频信息

    Args:
        input_seg_file:待统计的分词文件
        output_vocab_file:统计后的词频文件
    '''
    with open(input_seg_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split(' '):
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    # sorted_word_list = [('c', 2), ('b', 1), ('a', 0)]
    sorted_word_list = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    with open(output_vocab_file, 'w', encoding='utf-8') as f:
        f.write('<UNK>\t1000000\n')
        for word_item in sorted_word_list:
            f.write('{}\t{}\n'.format(word_item[0], word_item[1]))

def generate_category_file(input_file, category_file):
    '''
    统计类型信息

    Args:
        input_file:
        category_file:
    '''
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)
    with open(category_file, 'w', encoding='utf-8') as f:
        for category in category_dict:
            line = '{}\n'.format(category)
            print('{}:{}\n'.format(category, category_dict[category]))
            f.write(line)


if __name__ == "__main__":
    # generate_word_seg_file(train_file, segment_train_file)
    # generate_word_seg_file(test_file, segment_test_file)
    # generate_word_seg_file(val_file, segment_val_file)
    # generate_vocab_file(segment_train_file, vocab_file)
    generate_category_file(train_file, category_file)