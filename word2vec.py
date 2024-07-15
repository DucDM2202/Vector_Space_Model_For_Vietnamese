# -*- coding: utf-8 -*-
import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models import Word2Vec

# path data
pathdata = './datatrain.txt'

def read_data(path):
    traindata = []
    sents = open(pathdata, 'r', encoding='utf-8').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata


if __name__ == '__main__':
    train_data = read_data(pathdata)

    model = Word2Vec(train_data, vector_size=150, window=10, min_count=2, workers=4, sg=0)
    # current_dir = os.path.dirname(__file__)

    # # Đường dẫn tương đối đến thư mục bạn muốn lưu trữ mô hình
    # save_path = os.path.join(current_dir, "../model/word2vec_skipgram.model")

    # # Lưu mô hình Word2Vec
    # model.wv.save(save_path)
    model.wv.save("word2vec_skipgram.model")
