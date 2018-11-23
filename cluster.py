#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys
import os
import json
import math
import re
import pandas as pd
from pyhanlp import *
import numpy as np


def get_chunks(json_file):
    json_file_lines = open(json_file, 'r')
    depth1 = 2500
    depth2 = 2500
    depth = 2500
    index_1 = 2500
    index_2 = 2500
    idx = 2500

    json_lines = json.load(json_file_lines)
    depths = []
    dict_chunks = {}
    for term_dict in json_lines:
        pattern1 = u'.*您与我们.*的合同|.*关于本保险合同|第 1 章 您与我们的保险合同' \
                   u'|.*投保人与本公司的合同|1 被保险人范围|1 投保人与我们的合同' \
                   u'|.*您与本公司.*的合同|.*投保人与本公司订立的合同|第一章.*保险合同的构成|1 投保范围'

        if (re.match(pattern1, term_dict['text'])):
            depth1 = term_dict['depth']
            index_1 = term_dict['idx']
    for term_dict in json_lines:
        pattern2 = u'1\..*合同的?构成|第一条.*合同的?构成|1\.1.*合同的?构成' \
                   u'|第一章.*合同的?构成|一\、.*合同的?构成|第一条.*合同的?订立' \
                   u'|1.*被保险人范围|1\..*投保范围|1.*定义与释义|第一章.*定义' \
                   u'|1\.1.*合同的?订立|第一条.*委托管理基金|.*您与我们的?合同' \
                   u'|一、.*委托管理合同的构成|1.*投保人与我们的合同'
        if (re.match(pattern2, term_dict['text'])):
            depth2 = term_dict['depth']
            index_2 = term_dict['idx']
    if (depth1 != 2500):
        depth = depth1
        idx = index_1
    else:
        depth = depth2
        idx = index_2


    for term_dict in json_lines:
        if (term_dict['depth'] == depth):
            depths.append(term_dict['text'])
    for i, term in enumerate(depths):
        if (i <= len(depths) - 2):
            dict_chunks[term] = []
            flag = False
            index = 5000
            for j, dictt in enumerate(json_lines):
                if (j <= len(json_lines) - 2 and json_lines[j]['idx'] >= idx):
                    if (dictt['text'] == term):
                        index = j
                    while (j > index and json_lines[j]['text'] != depths[i + 1] and json_lines[j][
                        'depth'] > depth and j < len(json_lines) - 1):
                        flag = True
                        dict_chunks[term].append(json_lines[j]['text'])
                        j = j + 1
                    if (json_lines[j]['text'] == depths[i + 1]):
                        flag = True
                    if (flag):
                        break
    return dict_chunks

def write_chunks(json_path,chunks_path):
    files=os.listdir(json_path)
    for file in files:
        json_file=os.path.join(json_path,file)
        chunks_file = os.path.join(chunks_path, file)
        chunks_dict=get_chunks(json_file)
        if(os.path.exists(chunks_file)):
            os.remove(chunks_file)
        with open(chunks_file,'w') as f:
            for term in chunks_dict.keys():
                f.write(term.encode('utf-8')+'##')
                for term1 in chunks_dict[term]:
                    if('|' not in term1.encode('utf-8') and '-' not  in term1.encode('utf-8')
                            and  '....' not  in term1.encode('utf-8') and '--' not  in term1.encode('utf-8')
                            and len(term1.encode('utf-8'))<1000 ):
                        f.write(term1.encode('utf-8'))
                f.write('\n')


def seg(chunk_path,fenci_filename):
    files=os.listdir(chunk_path)
    with open(fenci_filename,'w') as f:
        for file in files:
            filename=os.path.join(chunk_path,file)
            file_lines=open(filename,'r')
            for line in file_lines:
                line_list=[]
                for term in HanLP.segment(line):
                    line_list.append(str(term.word))
                line_str=' '.join(line_list)
                f.write(line_str)


def remove_sign(fenci_sign, fenci):
    if(os.path.exists(fenci)):
        os.remove(fenci)
    f_lines = open(fenci_sign, 'r')
    with open(fenci, 'w') as f:
        for line in f_lines:
            line=line.replace('##',' ')
            f.write(line)


def tf_idf_extract(filename,filename_fenci):
    tf_dict={}
    f_lines=open(filename_fenci,'r')
    for line in f_lines:
        line_list=line.split(' ')
        for word in line_list:
            if(word not in tf_dict.keys()):
                tf_dict[word]=[1]
            else:
                tf_dict[word][0]=tf_dict[word][0]+1

    df=pd.read_excel(filename)
    frequency_table={}
    df = np.array(df).tolist()
    for line_list in df:
        frequency_table[line_list[1].encode('utf-8')]=float(line_list[3])

    for term in tf_dict.keys():
        tf_dict[term][0]=tf_dict[term][0]/len(tf_dict)
        try:
            frequency=frequency_table[term]
        except:
            frequency=float(7.7946)
            print('------')
        x=1.0/(1.01-((1.0-frequency/100.0)**5000))
        try:
            idf = math.log(x,2)
        except:
            print(frequency)
            print(x)
        tf_dict[term].append(idf)

    for term in tf_dict.keys():
        tf_dict[term].append(tf_dict[term][0]*tf_dict[term][1])

    tf_dict=sorted(tf_dict.items(),key=lambda item:item[1][2],reverse=True)
    tf_idf_dict = {}
    for i, term in enumerate(tf_dict):
        if (i < 200):
            tf_idf_dict[term[0]] = term[1][2]
    return tf_idf_dict


def tf_idf_word(sentence): # sentence is a string ,splitting by ' '
    sentence = sentence.split(' ')
    word2vec = {}
    frequency_dict = {}
    # tf_idf_dict=tf_idf_extract(filename_csv, filename_fenci)
    for term in tf_idf_dict.keys():
        frequency_dict[term] = 0
    for word in sentence:
        if word in tf_idf_dict.keys():
            if word in frequency_dict.keys():
                frequency_dict[word] += 1

    for term in tf_idf_dict.keys():
        try:
            word2vec[term] = tf_idf_dict[term] * frequency_dict[term]
        except:
            print('=======' + term + '=======')
            print(tf_idf_dict[term])
            print(frequency_dict[term])
    return word2vec


def sent2vec_tfidf_title(sentence):
    first_title=sentence.split('##')[0]
    text=''
    try:
        text=sentence.split('##')[1]
    except:
        print(sentence)
    tf_idf_title=tf_idf_word(first_title)
    tf_idf_text=tf_idf_word(text)
    tf_idf_dict={}
    for text_term in tf_idf_text.keys():
        tf_idf_dict[text_term]=tf_idf_text[text_term]+10*tf_idf_title[text_term]
    return tf_idf_dict

def write_tfidf_title(filename_fenci_sign,filename_sent2vec_title):
    if(os.path.exists(filename_sent2vec_title)):
        os.remove(filename_sent2vec_title)
    file_lines = open(filename_fenci_sign, 'r')
    with open(filename_sent2vec_title, 'w') as f:
        for line in file_lines:
            sent2vec_list=[]
            sent2vec = sent2vec_tfidf_title(line)
            for word in sent2vec.keys():
                sent2vec_list.append(str(sent2vec[word]))
            sent2vec_str = ' '.join(sent2vec_list)
            f.write(sent2vec_str + '\n')





if __name__ == "__main__":
    json_path=r'jsonn/'
    filename_fenci = r'fenci'
    filename_fenci_sign=r'fenci_sign'
    chunk_path = r'chunks/'
    filename_csv = r'frequency_table.xlsx'
    kmeans_result_tfidf_origin_path = r'kmeans_result_tfidf_origin/'
    kmeans_result_tfidf_fenci_path = r'kmeans_result_tfidf_fenci'
    filename_sent2vec_title=r'filename_sent2vec_title'

    write_chunks(json_path, chunk_path)
    seg(chunk_path, filename_fenci_sign)
    remove_sign(filename_fenci_sign, filename_fenci)
    tf_idf_dict = tf_idf_extract(filename_csv, filename_fenci)
    write_tfidf_title(filename_fenci_sign, filename_sent2vec_title)


