# -*- coding:utf-8 -*-

import codecs

loadfile = codecs.open("happiness_seg.txt", "r",encoding='utf-8')
#writefile = codecs.open("2.txt", "w",encoding='utf-8')

lists = loadfile.read().split(" ") # 把词存成数组
dicts = {} # 定义一个字典用来存放二元词组


def wordDict(text):
    '''筛选二元词组存成字典'''

    length = len(lists)
    for i in range(length-1):
        first_word = lists[i] #当前词
        second_word = lists[i+1] #下一个词

        if len(first_word) >= 2 and len(second_word) >= 2: #去掉单字词
            tuple_word = first_word + " " + second_word    #组成二元词组
            #print(tuple_word)
            if tuple_word not in dicts:
                dicts[tuple_word] = 1
            else:
                dicts[tuple_word] +=1 # 统计次数
                #print dicts[tuple_word]
    return  dicts


wordDict(lists)

# 用字典中 value 排序
sortedList = sorted(dicts.items(), key=lambda x: x[1], reverse=True)

print "---二元词组统计---"

for i in sortedList[:10]:
    print " '%s' appeared %d times" % (i[0], i[1])
