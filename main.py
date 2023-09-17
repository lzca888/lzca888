import re
import os
import jieba
# from line_profiler import LineProfiler
from memory_profiler import profile
from simhash import Simhash
# 空间性能测试
@profile
# jieba分词，把标点符号、转义符号等特殊符号过滤掉,只保留数字，大小写字母以及中文
def filter(str):
    # 创建了一个正则表达式模式
    remove_rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")  # 只保留数字，大小写字母以及中文
    str = remove_rule.sub('', str)
    text = jieba.lcut(str)  # 分词
    return text


# 读取文件内容
def get_file_contents(path):

    str = ''
    f = open(path, 'r', encoding='UTF-8')
    line = f.readline()
    while line:
        str = str + line
        line = f.readline()
    f.close()
    return str

@profile
def main(original, copy, result):
    str1 = get_file_contents(original)
    str2 = get_file_contents(copy)
    original_words = filter(str1)
    copy_words = filter(str2)

    # 生成Simhash值
    original_simhash = Simhash(original_words)
    copy_simhash = Simhash(copy_words)

    # 计算海明距离
    distance = original_simhash.distance(copy_simhash)
    # 计算查重率
    similarity = (1 - distance / 64) * 100
    f = open(result, 'w', encoding="utf-8")
    f.write('文章重复率为：{:.2f}%\n'.format(similarity))
    f.close()
    print('文章重复率为：{:.2f}%'.format(similarity))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path1 = input("输入论文原文的文件的绝对路径：")
    path2 = input("输入抄袭版论文的文件的绝对路径：")
    path3 = input("输入输出文件的绝对路径：")

    if not os.path.exists(path1):
        print("论文原文文件不存在！")
        exit()
    if not os.path.exists(path2):
        print("抄袭版论文文件不存在！")
        exit()
    if not os.path.exists(path3):
        print("输出文件不存在！")
        exit()

    main(path1, path2, path3)
    '''
    # 时间性能测试
    p = LineProfiler()
    p.add_function(get_file_contents)
    p.add_function(filter)
    test_func = p(main)
    test_func(path1, path2, path3)
    p.print_stats()  # 控制台打印相关信息
    '''