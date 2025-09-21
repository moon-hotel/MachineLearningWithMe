from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz


def load_data():
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names


def train(X_train, X_test, y_train, y_test, feature_names):
    model = tree.DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)
    print("在测试集上的准确率为：", model.score(X_test, y_test))
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=feature_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('iris')

    # 除了需要 pip install 安装 graphviz 包以外，本地也需要安装 graphviz程序
    # Mac: brew install graphviz
    # Ubuntu: sudo apt install graphviz
    # Centos: sudo yum install graphviz
    # Windows下载: https://graphviz.org/download/
    # 具体步骤参见《跟我一起学机器学习》第8.2.2节内容

    # 以下为 MacOS 安装 graphviz 步骤，window 忽略
    # Step 1. 则需要先下载brew工具
    # 地址：https://www.macports.org/install.php
    # 选择对应版本的pkg文件，然后双击安装
    # Step 2. 安装graphviz
    # sudo port install graphviz
    # 如果上面安装不成功，则可以以源码的方式进行安装，一下方式可以参考
    # 下载：https://github.com/macports/macports-base/releases/download/v2.7.1/MacPorts-2.7.1.tar.gz
    # 解压：tar zxvf MacPorts-2.7.1.tar.gz
    # cd MacPorts-2.7.1
    # ./configure && make && sudo make install
    # 然后再安装graphviz
    # sudo port install graphviz


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = load_data()
    train(X_train, X_test, y_train, y_test, feature_names)
