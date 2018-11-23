#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys
import os
from pyhanlp import *
import numpy as np

def remove_other(kmeans_result_tfidf_origin_path,kmeans_result_tfidf_path):
    files=os.listdir(kmeans_result_tfidf_origin_path)
    for file in files:
        filename=os.path.join(kmeans_result_tfidf_origin_path,file)
        file_lines=open(filename,'r')
        filename_write=os.path.join(kmeans_result_tfidf_path,file)
        if(os.path.exists(filename_write)):
            os.remove(filename_write)
        distances = []
        for i, line in enumerate(file_lines):
            if (i >= 3):
                distances.append(float(line.split()[0]))
        distances = distances[0:len(distances) // 2]

        file_lines = open(filename, 'r')
        with open(filename_write,'w') as f:
            for i, line in enumerate(file_lines):
                if(i>=3):
                    line_list=line.split()[1:]
                    distance=float(line.split()[0])
                    if(distance in distances):
                        line_str=' '.join(line_list)
                        line_listt=[]
                        for term in HanLP.segment(line_str):
                            line_listt.append(str(term.word))
                        line_strr = ' '.join(line_listt)
                        f.write(line_strr+'\n')

if __name__ == "__main__":
    kmeans_result_tfidf_origin_path = r'kmeans_result_tfidf_origin/'
    kmeans_result_tfidf_fenci_path = r'kmeans_result_tfidf_fenci'
    remove_other(kmeans_result_tfidf_origin_path, kmeans_result_tfidf_fenci_path)
