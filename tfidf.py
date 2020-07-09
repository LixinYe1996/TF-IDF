import os
import codecs
import math
import operator
from sklearn.datasets import load_files
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def fun(filepath):  # 遍历文件夹中的所有文件，返回文件list
    arr = []
    for root, dirs, files in os.walk(filepath):
        for fn in files:
            arr.append(root + "\\" + fn)
    return arr


def wry(txt, path):  # 写入txt文件
    f = codecs.open(path, 'a', 'utf8')
    f.write(txt)
    f.close()
    return path


def read(path):  # 读取txt文件，并返回list
    f = open(path, encoding="utf8")
    data = []
    for line in f.readlines():
        data.append(line)
    return data


def toword(txtlis):  # 将一片文章按照‘/’切割成词表，返回list
    wordlist = []
    alltxt = ''
    for i in txtlis:
        alltxt = alltxt + str(i)
    ridenter = alltxt.replace('\n', '')
    wordlist = ridenter.split(' ')
    return wordlist


def getstopword(path):  # 获取停用词表
    swlis = []
    for i in read(path):
        outsw = str(i).replace('\n', '')
        swlis.append(outsw)
    return swlis


def freqword(wordlis):  # 统计词频，并返回字典
    freword = {}
    for i in wordlis:
        if str(i) in freword:
            count = freword[str(i)]
            freword[str(i)] = count + 1
        else:
            freword[str(i)] = 1
    return freword


def wordinfilecount(vocabulary, corpuslist):  # 查出包含该词的文档数
    wordcount = {}
    for wordlist in corpuslist:
        # print(set(wordlist))
        # print(wordlist)
        for j in set(wordlist):
            if j in vocabulary:
                if j in wordcount.keys():
                    wordcount[j] = wordcount[j] + 1
                else:
                    wordcount[j] = 1
    print(wordcount)

    return wordcount


def tf_idf(wordlis, filelist, wordinfilecount):  # 计算TF-IDF,并返回字典
    # print(wordlis)
    # print(filelist)
    # print(corpuslist)
    outdic = {}
    tf = 0
    idf = 0
    dic = freqword(wordlis)
    # outlis = []
    for i in set(wordlis):
        if i in wordinfilecount.keys():
            # tf = dic[str(i)]/len(wordlis)  # 计算TF：某个词在文章中出现的次数/文章总词数
            tf = dic[str(i)]  # 计算TF：某个词在文章中出现的次数/文章总词数
            # 计算IDF：log(语料库的文档总数/(包含该词的文档数+1))
            idf = math.log((float(len(filelist)) + 1) / (wordinfilecount[str(i)] + 1) + 1)
            tfidf = tf * idf  # 计算TF-IDF
            # print(i,tf,idf,len(filelist),(wordinfilecount(str(i), corpuslist)),float(len(filelist))/(wordinfilecount(str(i), corpuslist)+1),tfidf)
            outdic[str(i)] = tfidf
    # L2范数归一化
    normx = 0
    for x in outdic.values():
        normx = normx + x ** 2
    normx = normx ** 0.5
    for w in outdic.keys():
        outdic[w] = outdic[w] / normx

    # orderdic = sorted(outdic.items(), key=operator.itemgetter(
    #     1), reverse=True)  # 给字典排序
    # print(type(orderdic))
    return outdic


def beindex(lis, vocabulary, i):
    row = np.array([])
    col = np.array([])
    data = np.array([])
    for word in lis:
        if word in vocabulary:
            row = np.append(row, i)
            col = np.append(col, vocabulary.index(word))
            data = np.append(data, float(lis[word]))
    return row, col, data


def befwry(lis, vocabulary):  # 写入预处理，将list转为string
    # outall = ''
    # for i in lis:
    #     ech = str(i).replace("('", '').replace("',", '\t').replace(')', '')
    #     outall = outall+'\t'+ech+'\n'
    outall = ''
    for word in vocabulary:
        if word in lis.keys():
            outall = outall + ',' + str(lis[word])
        else:
            outall = outall + ',' + str(0)
    outall = outall + '\n'
    return outall


if __name__ == '__main__':
    # main()
    import time

    start = time.time()

    n_features = 200

    news_data = load_files('D:/PycharmProjects/A2226_v325/5000SogouCS.corpus_seg/')
    data = (str(d, encoding="utf-8").split(' ') for d in news_data.data)
    print("summary: {0} documents in {1} categories.".format(len(news_data.data), len(news_data.target_names)))
    # news_train.target是长度为13180的一维向量，每个值代表相应文章的分类id
    print('news_categories_names:\n{}, \nlen(target):{}, target:{}'.format(news_data.target_names,
                                                                           len(news_data.target), news_data.target))

    corpuslist = []
    for d in data:
        corpuslist.append(d)
    filelist = corpuslist  # 获取文件列表

    vocabulary = {}
    for wordlist in filelist:
        for j in wordlist:
            if j in vocabulary:
                vocabulary[j] = vocabulary[j] + 1
            else:
                vocabulary[j] = 1

    wordinfilecount = wordinfilecount(vocabulary, filelist)

    # 特征词数
    vocabulary = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)  # 给字典排序
    print(vocabulary)

    features = {}
    for i in range(n_features):
        features[vocabulary[i][0]] = vocabulary[i][1]
    vocabulary = features
    print(vocabulary)
    X = []

    from scipy.sparse import csr_matrix

    row = np.array([])
    col = np.array([])
    data = np.array([])
    i = 0
    for wordlist in filelist:
        tfidfdic = tf_idf(wordlist, filelist, wordinfilecount)  # 计算TF-IDF
        r, c, d = beindex(tfidfdic, list(vocabulary.keys()), i)
        # print(index)
        row = np.append(row, r)
        col = np.append(col, c)
        data = np.append(data, d)
        i = i + 1

    X = sparse.csr_matrix((data, (row, col)), shape=(i, n_features))
    # for i in X:
    #     print(i)

    y = news_data.target
    from sklearn.model_selection import train_test_split

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    clf = MultinomialNB(alpha=0.0001)
    clf.fit(X, news_data.target)
    # # Create ROC curve
    # pred_probas = clf.predict_proba(X)[:, 1]  # score
    # fpr, tpr, _ = metrics.roc_curve(news_data.target, pred_probas)
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend(loc='lower right')
    # plt.savefig('NaiveBayes.png', bbox_inches='tight')

    print("predicting test dataset ...")

    pred = clf.predict(test_X)

    print("classification report on test set for classifier:")
    print(clf)
    print(news_data.target_names)
    print(classification_report(test_y, pred, target_names=news_data.target_names))
    print(news_data.target_names)

    cm = confusion_matrix(test_y, pred)
    print("confusion matrix:")
    print(cm)

    end = time.time()
    print("Execution Time: ", end - start)
