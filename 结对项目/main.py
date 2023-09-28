import jieba
import re
from simhash import Simhash
from memory_profiler import profile
from line_profiler import LineProfiler


# @profile()
def text_filter(text):
    # 将文本中的标点符号去除，得到纯净文本
    clean_text = re.sub(r'\W+', '', text).replace("_", '')
    # 使用jieba库的搜索引擎模式拆分纯净文本
    result = jieba.lcut_for_search(clean_text)
    # 返回拆分后的文本，形式为列表
    return result


# @profile()
def read_txt(path):
    # 以读文件方式打开路径中的文件
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
            if not text:
                # 抛出异常
                raise FileNotFoundError(path, "file is empty")
    except FileNotFoundError as e:
        print(f"获取文件失败:{e}")
    else:
        # 将文件内容过滤为纯净文本并拆分，为simhash做准备
        result = text_filter(text)
        return result


# @profile()
def check_similarity(org, copy, ans):
    org_text = read_txt(org)
    copy_text = read_txt(copy)

    # 通过simhash中的Simhash方法计算出文本的simhash值
    try:
        org_hash = Simhash(org_text)
        if not org_hash:
            raise Exception("读取失败")
        copy_hash = Simhash(copy_text)
        if not copy_hash:
            raise Exception("读取失败")

        # 计算出原文和抄袭文本simhash的海明距离
        distance = org_hash.distance(copy_hash)
        # 通过海明距离来计算相似度 64是因为simhash特征向量长度为64
        similarity = 1 - (distance / 64)
        print("similarity:{:.2f}%".format(similarity * 100))
        # 将重复率写入文件ans.txt中 保留两位小数
        with open(ans, 'w', encoding='utf-8') as file:
            file.write("重复率为：{:.2f}%".format(similarity * 100))
    except Exception:
        print("simhash计算失败")


if __name__ == '__main__':
    org_path = input("请输入论文原文的文件的绝对路径：")
    copy_path = input("请输入抄袭版论文的文件的绝对路径：")
    ans_path = input("请输入输出的答案文件的绝对路径：")
    # 运行主程序check_similarity()的语句 测试内存占用时使用 测试时间占用时需要注释
    check_similarity(org_path, copy_path, ans_path)

    # 时间性能分析
    '''
    lp = LineProfiler()
    lp.add_function(text_filter)
    lp.add_function(read_txt)
    test_func = lp(check_similarity)
    test_func(org_path, copy_path, ans_path)
    lp.print_stats()
    '''
