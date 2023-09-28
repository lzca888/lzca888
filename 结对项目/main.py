import random
from fractions import Fraction
from line_profiler import LineProfiler
from memory_profiler import profile
import sys
import getopt

class Fractions(object):
    def __init__(self):
        self.numerator = None
        self.denominator = None

    def setNumerator(self,numerator):
        self.numerator = numerator

    def setDenominator(self,denominator):
        self.denominator = denominator

    def toString(self):
        a = self.numerator
        b = self.denominator
        return str(Fraction(a, b))



class Create(object):
    # 创建四则运算符号
    def create_operator(self):
        operator = ["+", "-", "×", "÷"]
        return random.choice(operator)

    # 生成四则运算表达式
    def create_arith(self, r):
        x = 0
        list = []
        # 随机生成运算符的数量1-3
        operator_num = random.randint(1, 3)
        e1 = Create()
        e2 = Create()
        if operator_num == 1:
            list.append(e1.create_number(r))
            list.append(e2.create_operator())
            list.append(e1.create_number(r))
        elif operator_num == 2:
            start = random.randint(0, 2)
            end = 0
            if start > 0:
                end = start + 1
            for i in range(1, 4):
                if i == start:
                    list.append("(")
                list.append(e1.create_number(r))
                if i == end:
                    list.append(")")
                list.append(e2.create_operator())
            list.pop()
        elif operator_num == 3:
            start = random.randint(0, 3)
            end = 0
            if start > 0:
                end = start + 1 + random.randint(0, 1)
                if end >= 4:
                    end = 4
            for i in range(1, 5):
                if i == start:
                    list.append("(")
                list.append(e1.create_number(r))
                if i == end:
                    list.append(")")
                list.append(e2.create_operator())
            list.pop()
        else:
            list.append(e1.create_number(r))
            list.append(e2.create_operator())
            list.append(e1.create_number(r))
        return list

    # 将表达式假分数转化为带分数
    def proper_fraction(self, list):
        num = 0
        for fract in list:
            if type(fract) == Fraction:
                n1 = fract.numerator
                n2 = fract.denominator
                if n2 == 1:
                    num += 1
                    continue
                elif n1 > n2:
                    sub = int(n1/n2)
                    n1 = n1 % n2
                    list[num] = '%d%s%d/%d' %(sub, '’', n1,n2)
            num += 1
        return list

    # 将答案假分数转化为带分数
    def pop_fracte(self, re):
        n1 = re.numerator
        n2 = re.denominator
        if n2 == 1:
            return n1
        elif n1 < n2:
            return re
        else:
            sub = int(n1/n2)
            n1 = n1 % n2
            return '%d%s%d/%d' % (sub, '’', n1, n2)

    # 生成随机数
    def create_number(self, r):
        b = random.randint(1, r)
        a = random.randint(1, b * r)
        n = Fraction(a, b)
        return n

    # 将列表中的表达式转化成字符串
    def to_string(self, list):
        np = ""
        for i in range(len(list)):
             np = np + str(list[i])
        return np


class Judge(object):

    # 生成逆波兰式
    def toRPN(self, list):
        right = []
        aStack = []
        position = 0
        while True:
            if self.isOperator(list[position]):
                if list == [] or list[position] == "(":
                    aStack.append(list[position])
                else:
                    if list[position] == ")":
                        while True:
                            if aStack != [] and aStack[-1] != "(":
                                operator = aStack.pop()
                                right.append(operator)
                            else:
                                if aStack != []:
                                    aStack.pop()
                                break
                    else:
                        while True:
                            if aStack != [] and self.priority(list[position], aStack[-1]):
                                operator = aStack.pop()
                                if operator != "(":
                                    right.append(operator)
                            else:
                                break
                        aStack.append(list[position])
            else:
                right.append(list[position])
            position = position + 1
            if position >= len(list):
                break
        while aStack != []:
            operator = aStack.pop()
            if operator != "(":
                right.append(operator)
        return right

    #  将逆波兰式转换成二叉树并规范化
    def createTree(self, suffix):
        stacks = []

        for i in range(0, len(suffix)):
            tree = BinaryTree()
            ob = suffix[i]
            c = Caculation()
            if self.isOperator(ob):
                t2 = BinaryTree()
                t1 = BinaryTree()
                t2 = stacks.pop()
                t1 = stacks.pop()
                if ob == '-' and t1.value <= t2.value:
                    return None
                else:
                    if self.maxTree(t1, t2):
                        tree.set_date(ob)
                        tree.set_left(t1)
                        tree.set_right(t2)
                        tree.set_value(c.caulate(ob, t1.value, t2.value))
                    else:
                        tree.set_date(ob)
                        tree.set_left(t2)
                        tree.set_right(t1)
                        tree.set_value(c.caulate(ob, t1.value, t2.value))
                    stacks.append(tree)
            else:
                tree.set_value(ob)
                tree.set_date(ob)
                stacks.append(tree)
        return tree

    # 判断是否为符号
    def isOperator(self, operator):
        if operator == "+" or operator == "-" or operator == "×" or operator == "÷" or operator == "(" or operator == ")":
            return True
        else:
            return False

    #  当两颗树value值相等时判定优先级
    def priority(self, operatorout, operatorin):
        m = -1
        n = -1
        addop = [["+", "-", "×", "÷", "(", ")"], ["+", "-", "×", "÷", "(", ")"]]
        first = [[1, 1, 2, 2, 2, 0], [1, 1, 2, 2, 2, 0],
                 [1, 1, 1, 1, 2, 0], [1, 1, 1, 1, 2, 0],
                 [2, 2, 2, 2, 2, 0], [2, 2, 2, 2, 2, 2]]
        for i in range(6):
            if operatorin == addop[0][i]:
                m = i
        for i in range(6):
            if operatorout == addop[1][i]:
                n = i
        if m == -1 and n == -1:
            return False
        elif m == -1 and n != -1:
            return False
        elif m != -1 and n == -1:
            return True
        elif first[m][n] == 1:
            return True
        else:
            return False

    # 判断左右子树
    def maxTree(self, t1, t2):
        c = Caculation()
        max = c.max(t1.value, t2.value)  # 比较两个树value值大小
        if max == 1:
            return True
        elif max == 2:
            return False
        elif self.priority(t1.date, t2.date):  # 如果两个树的value值相等，则判定优先级
            if t1.left == None or t2.left == None:
                return True
            max = c.max(t1.left.value, t2.left.value)
            if max == 1:
                return True
            elif max == 2:
                return False
            else:
                return True
        return False


class Compare(object):

    #  对两个文件中的答案进行比较并记录
    def grade(self, reanswer_file, answer_file):
        correct = []
        wrong = []
        co = 0
        wr = 0
        with open(answer_file, 'r', encoding='utf-8') as f1, open(reanswer_file, 'r', encoding='utf-8') as f2:
            answers = f2.readlines()
            line = 0
            for r_answers in f1.readlines():
                if answers[line] == r_answers:
                    co += 1
                    correct.append(line+1)
                else:
                    wr += 1
                    wrong.append(line+1)
                line += 1
        with open('Grade.txt', 'w') as f3:
            f3.write(f"Correct: {str(co)} ({', '.join(str(s) for s in correct if s not in [None])})" + '\n')
            f3.write(f"Wrong: {str(wr)} ({', '.join(str(s) for s in wrong if s not in [None])})" + '\n')
        print("文件比较完成")



class Check(object):

    def __init__(self):
        self.check = []

    #  对二叉树进行判重
    def check_tree(self, tree):
        if self.check == []:
            self.check.append(tree)
            return True
        else:
            for i in range(len(self.check)):
                if self.check[i] == tree:
                    return False
        self.check.append(tree)
        return True


class Caculation(object):

    #  计算相应运算符下两参数的值
    def caulate(self, op, f1, f2):
        result = Fractions()
        n1 = int(f1.numerator)
        d1 = int(f1.denominator)
        n2 = int(f2.numerator)
        d2 = int(f2.denominator)
        list = []
        if op == '+':
            re = Fraction(n1, d1) + Fraction(n2, d2)

        elif op == '-':
            re = Fraction(n1, d1) - Fraction(n2, d2)

        elif op == '×':
            re = Fraction(n1, d1) * Fraction(n2, d2)

        else:
            re = Fraction(n1, d1) / Fraction(n2, d2)

        return re

    # 比较传入参数大小并返回
    def max(self, num1, num2):
        n1 = int(num1.numerator)
        d1 = int(num1.denominator)
        n2 = int(num2.numerator)
        d2 = int(num2.denominator)
        m1 = n1 * d2
        m2 = n2 * d1
        if m1 > m2:
            return 1
        elif m1 < m2:
            return 2
        else:
            return 3



# 二叉树
class BinaryTree(object):
    def __init__(self):
        self.date = None
        self.left = None
        self.right = None
        self.value = None

    def tree(self, date, left, right, value):
        self.date = date
        self.left = left
        self.right = right
        self.value = value

    def set_date(self, date):
        self.date = date

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def set_value(self, value):
        self.value = value

    def to_string(self, tree):
        s = ""
        s = self.out_put_tree(tree, s)
        return s

    def out_put_tree(self, tree, s):
        if tree != None:
            s1 = self.out_put_tree(tree.left, s)
            s2 = self.out_put_tree(tree.right, s)
            if type(tree.date) == Fractions:
                return str(s1) + str(s2) + str(tree.date.to_string())
            else:
                return str(s1) + str(s2) + str(tree.date)
        return s


class Arith(object):

    # 生成问题和答案在判重后写入文件
    def creat(self, problem_number, r):
        creat_pro = Create()
        t = BinaryTree()
        c = Check()
        with open("Exercises.txt", "w", encoding='utf-8') as file1, open("Answer.txt", "w", encoding='utf-8') as file2:
            num = 0
            while num < problem_number:
                arith = creat_pro.create_arith(r)  # 生成四则运算列表
                Ju = Judge()
                al = Ju.toRPN(arith)  # 将列表转换成逆波兰式
                string = creat_pro.to_string(creat_pro.proper_fraction(arith))
                ta = Ju.createTree(al)  # 将逆波兰式生成规范二叉树
                if ta:
                    val = str(creat_pro.pop_fracte(ta.value))
                    if c.check_tree(t.to_string(ta)):  # 进行判重
                        file1.write("%d. " % (num+1) + string + '\n')
                        file2.write("%d. " % (num+1) + val + '\n')
                        num +=1
        print("四则运算题目生成完毕，数量为%d个" % problem_number)

    # 支持命令行键入参数
    # @profile()
    def main(self, arith, argv):
        problem_number = None
        num_range = None
        exercise_file = None
        answer_file = None
        try:
            opts, args = getopt.getopt(argv, "n:r:e:a:")
        except getopt.GetoptError:
            print('Error: main.py -n <problem_number> -r <num_range>')
            print('   or: main.py -e <ReAnswer_file>.txt -a <Answer_file>.txt')
            sys.exit(2)

        for opt, arg in opts:
            if opt in("-n"):
                problem_number = int(arg)
            elif opt in ("-r"):
                num_range = int(arg)
            elif opt in("-e"):
                exercise_file = arg
            elif opt in("-a"):
                answer_file = arg
        if problem_number and num_range:
            arith.creat(problem_number, num_range)
        elif exercise_file and answer_file:
            compare = Compare()
            compare.grade('ReAnswer.txt', 'Answer.txt')
        else:
            print('Error: main.py -n <problem_number> -r <num_range>')
            print('   or: main.py -e <ReAnswer_file>.txt -a <Answer_file>.txt')



if __name__ == "__main__":

    arith = Arith()
    compare = Compare()
    lp = LineProfiler()  # 构造分析对象
    # arith.main(arith, sys.argv[1:])
    lp.add_function(compare.grade)
    test_func = lp(arith.main)  # 添加分析主函数，注意这里并不是执行函数，所以传递的是是函数名，没有括号。
    test_func(arith, sys.argv[1:])  # 执行主函数，如果主函数需要参数，则参数在这一步传递，例如test_func(参数1, 参数2...)
    lp.print_stats()  # 打印分析结果
