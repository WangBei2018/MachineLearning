# coding:utf-8
import os
import jieba
import re
import sys
from functools import reduce
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import importlib
import numpy as np
import math
importlib.reload(sys)

def word_split(text):
    word_list = []
    pattern = re.compile(u'[\u4e00-\u9fa5]+')
    jieba_list = list(jieba.cut(text))
    time = {}
    for i, c in enumerate(jieba_list):
        if c in time:  # record appear time
            time[c] += 1
        else:
            time.setdefault(c, 0) != 0
        if pattern.search(c):  # if Chinese
            word_list.append((len(word_list), (text.index(c, time[c]), c)))
            continue
        if c.isalnum():  # if English or number
            word_list.append((len(word_list), (text.index(c, time[c]), c.lower())))  # include normalize
    return word_list

def word_index(text):
    words = word_split(text)
    #words = words_cleanup(words)
    return words

def inverted_index(text):
    inverted = {}

    for index, (offset, word) in word_index(text):
        locations = inverted.setdefault(word, [])
        locations.append((index, offset))

    return inverted

def inverted_index_add(inverted, doc_id, doc_index):
    for word, locations in doc_index.items():
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted

def search(inverted, query):
    words = [word for _, (offset, word) in word_index(query) if word in inverted]  # query_words_list
    results = [set(inverted[word].keys()) for word in words]
    # x = map(lambda old: old+1, x) 
    doc_set = reduce(lambda x, y: x & y, results) if results else []
    precise_doc_dic = {}
    if doc_set:
        for doc in doc_set:
            index_list = [[indoff[0] for indoff in inverted[word][doc]] for word in words]
            offset_list = [[indoff[1] for indoff in inverted[word][doc]] for word in words]

            precise_doc_dic = precise(precise_doc_dic, doc, index_list, offset_list, 1)  # 词组查询
            precise_doc_dic = precise(precise_doc_dic, doc, index_list, offset_list, 2)  # 临近查询
            precise_doc_dic = precise(precise_doc_dic, doc, index_list, offset_list, 3)  # 临近查询

        return precise_doc_dic
    else:
        return {}

def precise(precise_doc_dic, doc, index_list, offset_list, range):
    if precise_doc_dic:
        if range != 1:
            return precise_doc_dic  # 如果已找到词组,不需再进行临近查询
    phrase_index = reduce(lambda x, y: set(map(lambda old: old + range, x)) & set(y), index_list)
    phrase_index = list(map(lambda x: x - len(index_list) - range + 2, phrase_index))

    if len(phrase_index):
        phrase_offset = []
        for po in phrase_index:
            phrase_offset.append(offset_list[0][index_list[0].index(po)])  # offset_list[0]代表第一个单词的字母偏移list
        precise_doc_dic[doc] = phrase_offset
    return precise_doc_dic


if __name__ == '__main__':

    inverted = {}
    documents = {}
    corpus = []
    for filename in os.listdir('data'):
        f = open('data//' + filename,encoding="UTF-8").read()
        documents.setdefault(filename.encode('utf-8').decode('utf-8'), f)
    for doc_id, text in documents.items():
        corpus.append(list(jieba.cut(text)))
        #print('---------------',list(jieba.cut(text)))
        doc_index = inverted_index(text)
        inverted_index_add(inverted, doc_id, doc_index)

    for word, doc_locations in inverted.items():
        print (word, doc_locations)


    spilt_str = []
    for i in range(0,len(corpus)):
        a = (" ".join(str(i) for i in corpus[i]))
        #print(a)
        spilt_str.append(a)                         #分词后的字符数组
    print(spilt_str)
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(spilt_str)
    print('count ：',count)
    words = vectorizer.get_feature_names()
    print('------------',words)
    D = count.todense()
    print(D)
    # Search something and print results
    # queries = ['创始人', '谷歌', '加盟']
    queries = words
    docnum = []
    for query in queries:
        result_docs = search(inverted, query)
        print("Search for '%s': %s" % (query, u','.join(result_docs.keys())))   # %s是str()输出字符串%r是repr()输出对象
        #print('******************88',len(result_docs.keys()))
        docnum.append(len(result_docs.keys()))
        def extract_text(doc, index):
            return documents[doc].encode('utf-8').decode('utf-8')[index:index + 30].replace('\n', ' ')

        if (result_docs):
            for doc, offsets in result_docs.items():
                for offset in offsets:
                    print ('   - %s' % extract_text(doc, offset))
        else:
                    print ('Nothing found!')
    #print('---corpus:',corpus)

    #将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(spilt_str)
    print(count)
    #该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    #第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(spilt_str))
    print('========',vectorizer.fit_transform(spilt_str))
    words = vectorizer.get_feature_names()
    print('------------',words)
    print(tfidf)
    weight = tfidf.toarray()
    print(weight)