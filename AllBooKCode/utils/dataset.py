import os
import re

data_home = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def clean_str(string, sep=" "):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param string: 输入必须是字符串类型
    :param sep: 表示去掉的部分用什么填充，默认为一个空格
    :return: 返回处理后的字符串
    example:
    s = "祝你2018000国庆快乐！"
    print(clean_str(s))# 祝你 国庆快乐
    print(clean_str(s,sep=""))# 祝你国庆快乐
    """
    string = re.sub(r"[^\u4e00-\u9fff]", sep, string)
    string = re.sub(r"\s{1,}", sep, string)  # 若有空格，则最多只保留1个宽度
    return string.strip()


def load_spam():
    data_spam_dir = os.path.join(data_home, 'spam')

    def load_spam_data(file_path=None):
        texts = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                texts.append(clean_str(line))
        return texts

    x_pos = load_spam_data(file_path=os.path.join(data_spam_dir, 'ham_5000.utf8'))
    x_neg = load_spam_data(file_path=os.path.join(data_spam_dir, 'spam_5000.utf8'))
    y_pos, y_neg = [1] * len(x_pos), [0] * len(x_neg)
    x, y = x_pos + x_neg, y_pos + y_neg
    return x, y
