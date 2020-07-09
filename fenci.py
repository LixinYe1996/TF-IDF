import jieba
import os
from threading import Thread, Lock
from queue import Queue
import codecs, sys, string, re

userdict = './dict.txt'
jieba.load_userdict(userdict)
LOCK = Lock()
stopkey = [w.strip() for w in codecs.open('stopWord.txt', 'r', encoding='utf-8').readlines()]


def readfile(filepath, encoding='utf-8'):
    # 读取文本
    with open(filepath, "rt", encoding=encoding) as fp:
        content = fp.read()
    return content


def savefile(savepath, content):
    # 保存文本
    with open(savepath, "wt", encoding='utf-8') as fp:
        fp.write(content)


def check_dir_exist(dir):
    # 坚持目录是否存在，不存在则创建
    if not os.path.exists(dir):
        os.mkdir(dir)


def text_segment(q):
    """
    对一个类别目录下进行分词
    """
    while not q.empty():
        from_dir, to_dir = q.get()
        with LOCK:
            print(from_dir)
        files = os.listdir(from_dir)
        for name in files:
            if name.startswith('.DS'):
                continue
            from_file = os.path.join(from_dir, name)
            to_file = os.path.join(to_dir, name)
            content = readfile(from_file)
            content = clearTxt(content)
            segList = jieba.cut(content, cut_all=False)
            segSentence = ''
            for word in segList:
                if word != '\t':
                    segSentence += word + " "
            seg_content = delstopword(segSentence)
            savefile(to_file, ''.join(seg_content))


def clearTxt(line):
    if line != '':
        line = line.strip()
        intab = string.punctuation + string.digits
        outtab = ""
        for i in range(len(intab)):
            outtab = outtab + ' '
        trantab = str.maketrans(intab, outtab)

        line = line.translate(trantab)
        # 去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        # 去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "", line)
    return line


def delstopword(line):
    wordList = line.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    # print(sentence)
    return sentence.strip()


def corpus_seg(curpus_path, seg_path):
    """对文本库分词，保存分词后的文本库,目录下以文件归类 curpus_path/category/1.txt, 保存为 seg_path/category/1.txt"""
    check_dir_exist(seg_path)
    q = Queue()
    cat_folders = os.listdir(curpus_path)
    for folder in cat_folders:
        from_dir = os.path.join(curpus_path, folder)
        to_dir = os.path.join(seg_path, folder)
        check_dir_exist(to_dir)

        q.put((from_dir, to_dir))

    for i in range(len(cat_folders)):
        Thread(target=text_segment, args=(q,)).start()


if __name__ == '__main__':
    # 分词
    corpus_seg('./5000SogouCS.corpus/', './5000SogouCS.corpus_seg/')
