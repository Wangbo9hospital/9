# 这是一个示例 Python 脚本。
import csv
import os
from mailcap import show

from narwhals import read_csv
from numpy import ndarray
from numpy.ma.core import shape
from openxlab.model.common.constants import model_cache_path
from torch.sparse import softmax


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。





#访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm


from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
roc

train_y, train_x = read_csv_columns('D:/R/文章全部数据/kqtrain_6vars.csv')
test_y, test_x = read_csv_columns('D:/R/文章全部数据/kqtest_6vars.csv')
test_y, test_x = read_csv_columns('D:/R/文章全部数据/kqval_6vars.csv')
clf = TabPFNClassifier(ignore_pretraining_limits=True,device='cpu',model_path='D:\\R\\model\\models\\models\\tabpfn-v2.5-classifier-v2.5_default.ckpt')  # Uses TabPFN 2.5 weights, finetuned on real data.

# To use TabPFN v2:
#clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
train_x=train_x[:,[0,2,4,5,6,8,14,15,16,18,20,21,24,25,27]]
test_x=test_x[:,[0,2,4,5,6,8,14,15,16,18,20,21,24,25,27]]
clf.fit(train_x, train_y)

# Predict probabilities
prediction_probabilities = clf.predict_proba(test_x)
print("ROC AUC:", roc_auc_score(test_y, prediction_probabilities[:, 1]))
np.savetxt("D:/R/tabpfn_pred_6vars_test.csv",prediction_probabilities,delimiter=",")

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))



import csv
import numpy as np


def read_csv_columns(file_path):
    """读取CSV文件，将第一列和其他列分离"""
    first_column = []
    other_columns = []

    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)

        # 跳过表头
        headers = next(csv_reader)
        print(f"表头: {headers}")

        # 读取数据
        for row in csv_reader:
            if row:  # 跳过空行
                first_column.append(row[0])  # 第一列
                other_columns.append(row[1:])  # 其余列
    first_column = np.array(first_column)
    other_columns = np.array(other_columns)
    return first_column, other_columns

def convert(string_2d):
    """
    使用NumPy将二维字符串数组转换为数值数组
    返回NumPy数组，处理非数值为NaN
    """
    # 将字符串列表转换为NumPy数组（仍然是字符串类型）
    str_array = np.array(string_2d, dtype=object)

    # 转换为浮点数，无法转换的变为NaN
    numeric_array = np.vectorize(
        lambda x: float(x) if isinstance(x, str) and
                              x.replace('.', '', 1).replace('-', '', 1).isdigit() else np.nan
    )(str_array)

    return numeric_array

# 使用示例
train_y, train_x = read_csv_columns('D:\\R\\model\\kqtrain_15vars.csv')
test_y, test_x = read_csv_columns('D:\\R\\model\\kqtest_15vars.csv')
train_x = convert(train_x)
test_x = convert(test_x)
train_y = convert(train_y)
test_y = convert(test_y)
train_y=[int(x) for x in train_y]
test_y=[int(x) for x in test_y]
train_y=np.array(train_y)
test_y=np.array(test_y)

train_x=train_x[0:682,:]
train_y=train_y[0:682]
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# 使用sklearn的permutation_importance
result = permutation_importance(
    clf, test_x, test_y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# 获取重要性分数
importance_scores = result.importances_mean
importance_std = result.importances_std

# 创建特征重要性DataFrame
import pandas as pd
feature_importance_df = pd.DataFrame({
    #'feature': feature_names,
    'importance': importance_scores,
    'std': importance_std
}).sort_values('importance', ascending=False)

print(feature_importance_df)

# 原始数组
arr=train_x

new_order = [1,14,10,8,13,11,12,4,0,9,5,6,7,3,2]


# 转置数组
transposed = list(zip(*arr))

# 按顺序选择列
selected_columns = [transposed[i] for i in new_order]

# 重新转置回来
rearranged = list(zip(*selected_columns))
train_xtop15 = np.array(rearranged)
clf = TabPFNClassifier(
        model_path='D:\\R\\model\\models\\models\\tabpfn-v2.5-classifier-v2.5_default.ckpt',ignore_pretraining_limits=True)

i=16
for i in range(1,16):
    train_xi=train_xtop15[:,0:i]
    test_xi=test_xtop15[:,0:i]
      # Uses TabPFN 2.5 weights, finetuned on real data.
    # To use TabPFN v2:
    # clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    clf.fit(train_xi, train_y)

    # Predict probabilities
    prediction_probabilities = clf.predict_proba(test_xi)
    print("ROC AUC:", roc_auc_score(test_y, prediction_probabilities[:, 1]))

#######################################################################################################################
#20fold，全部变量
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings('ignore')

y, X = read_csv_columns('D:\\R\\model\\fold10.csv')

X = convert(X)
y = convert(y)
y=[int(x) for x in y]
y=np.array(y)

clf = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True,model_path='D:\\R\\model\\models\\models\\tabpfn-v2.5-classifier-v2.5_default.ckpt')

###
X = X[:,[0,2,4,5,6,8,14,15,16,18,20,21,24,25,27]]
X = X[:,[2,27,20,15,0,24]]
###
# 创建分层K折交叉验证（保持类别比例）
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

all_true_labels = []
all_pred_probs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/20")

    # 划分训练集和验证集
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测

    y_proba = clf.predict_proba(X_val)

    all_true_labels.append(y_val)

    all_pred_probs.append(y_proba)

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np


# 假设 cv_results 是方法一的输出
# all_true_labels 是一维数组 true_labels_combined

def convert_to_onehot(labels, n_classes=None):
    """
    将标签转换为 one-hot 编码的二维数组

    参数:
    - labels: 一维标签数组
    - n_classes: 类别数量，如果为None则自动推断

    返回:
    - onehot_array: 二维one-hot编码数组
    """
    # 方法1: 使用sklearn的LabelBinarizer
    binarizer = LabelBinarizer()
    if n_classes is not None:
        # 确保所有类别都在标签中
        binarizer.fit(range(n_classes))
    else:
        binarizer.fit(labels)

    onehot_array = binarizer.transform(labels)
    return onehot_array

# 使用示例
true_labels = convert_to_onehot(all_true_labels)
true_labels = np.transpose(true_labels)
merged_array, second_columns = extract_and_merge_columns(all_pred_probs)
pred_probas = merged_array

np.savetxt('D:/R/kq/true_labels.csv', true_labels, delimiter=',')
np.savetxt('D:/R/kq/pred_probas.csv', pred_probas, delimiter=',')
np.savetxt('D:/R/kq/7vars_tabpfn.csv', fold20_7vars, delimiter=',')

def save_dict_list_to_csv(data, filename):
    """
    将字典列表保存为CSV文件
    """
    if not data:
        print("数据为空")
        return

    # 获取所有键作为列名
    fieldnames = data[0].keys()

    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # 写入列名
        writer.writerows(data)
    print(f"数据已保存到 {filename}")


import numpy as np




# 假设我们有10个这样的数组
# 这里只展示3个，你可以用你自己的10个数组

def extract_and_merge_columns(arr_list):
    """
    提取每个数组的第二列并合并

    参数:
    - arr_list: 列表，包含多个二维数组，每个形状为 (n, 2)

    返回:
    - merged_array: 合并后的二维数组
    """
    # 1. 提取每个数组的第二列
    second_columns = []

    for i, arr in enumerate(arr_list):
        # 检查数组形状
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"警告: 第{i}个数组形状为{arr.shape}，需要至少2列")
            continue

        # 提取第二列（索引为1）
        second_col = arr[:, 1]  # 这是一维数组
        second_columns.append(second_col)



        merged_array = np.column_stack(second_columns)

    return merged_array, second_columns


########################################################################################################################
import torch
print(torch.cuda.is_available())          # 应输出 True
print(torch.cuda.get_device_name(0))
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
#shap 分析
import torch

print(torch.__version__)
print(torch.cuda.is_available())

from tabpfn_extensions import interpretability
from tabpfn_extensions import TabPFNClassifier
import pandas as pd
import numpy as np
from tabpfn_extensions.interpretability.shap import get_shap_values, plot_shap

train_y, train_x = read_csv_columns('D:/3.29/kqtrain_15vars.csv')
test_y, test_x = read_csv_columns('D:/3.29/kqtest_15vars.csv')
train_x = convert(train_x)
test_x = convert(test_x)
train_y = convert(train_y)
test_y = convert(test_y)
train_y=[int(x) for x in train_y]
test_y=[int(x) for x in test_y]
train_y=np.array(train_y)
test_y=np.array(test_y)

clf = TabPFNClassifier(device='cuda', ignore_pretraining_limits=True,model_path='C:/tabpfn-v2.5-classifier-v2.5_default.ckpt')
clf.fit(train_x, train_y)
shap_values = get_shap_values(clf, test_x)

values_class1=shap_values.values[:,:,1]
base_values=shap_values.base_values
data=shap_values.data
np.savetxt('C:/base_values.csv', base_values, delimiter=',')
np.savetxt('C:/data.csv', data , delimiter=',')
np.savetxt('C:/values_class1.csv', values_class1 , delimiter=',')

plot_shap(shap_values)

import pickle
import numpy as np



# 方法A：保存多个变量到一个文件（推荐）
data_to_save = {
    'shap_values': shap_values,
    'classifier': clf,
    'train_x': train_x,
    'train_y': train_y,
    'test_x': test_x,
    'test_y': test_y,
    }


# 保存到文件
with open('C:/tabpfn_shap_15vars.pkl', 'wb') as f:  # 'wb'表示二进制写入
    pickle.dump(data_to_save, f)
print("✅ 所有工作已保存到 'tabpfn_analysis_session.pkl'")

with open('E:/tabpfn_shap_15vars.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

###################################################################################################################
#20fold,逐个增加变量
y, X = read_csv_columns('D:\\R\\model\\kqtrain_15vars.csv')

X = convert(X)
y = convert(y)
y=[int(x) for x in y]
y=np.array(y)
Xtop15 = X[0:2040,[1,14,10,8,13,11,12,4,0,9,5,6,7,3,2]] # 按重要性重新排列变量
Xtop15 = X[0:2040,[1,14,10,8,0,11,12,3,9,13,4,5,6,8,2]] # 按shap排序
y=y[0:2040]

fold15_auc=np.zeros(shape=(15,20))
i=15
for i in range(1,16):
    # 创建分层K折交叉验证（保持类别比例）
    X=Xtop15[:,0:i]
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"{i} vars, Fold {fold + 1}/20")

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_val)

        # 预测
        fold15_auc[i-1, fold] = roc_auc_score(y_val, prediction_probabilities[:, 1])
np.savetxt('D:/R/fig/3.5models/fold15_auc.csv', fold15_auc, delimiter=',')

X=X[:,[2,27,20,15,0,24]]
fold20_7vars=np.zeros(shape=(20))
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"var{7}, Fold {fold + 1}/20")

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_val)

        #预测
        fold20_7vars[fold] = roc_auc_score(y_val, prediction_probabilities[:, 1])

#######################################################################################################################


fold20=np.zeros(shape=(20))
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"var{6}, Fold {fold + 1}/20")

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_val)

        #预测
        fold20_atm09[fold] = roc_auc_score(y_val, prediction_probabilities[:, 1])
np.savetxt('D:/R/kq/atm09/fold20.csv', fold20, delimiter=',')

###################################################################################################################
#验证集,逐个增加变量
train_y, train_x = read_csv_columns('D:/R/model/kqtrain_15vars.csv')
test_y, test_x = read_csv_columns('D:/R/model/kqtest_15vars.csv')
train_x = convert(train_x)
test_x = convert(test_x)
train_y = convert(train_y)
test_y = convert(test_y)
train_y=[int(x) for x in train_y]
test_y=[int(x) for x in test_y]
train_y=np.array(train_y)
test_y=np.array(test_y)
train_x=train_x[:,[1,14,10,8,0,11,12,3,9,13,4,5,6,8,2]]
test_x=test_x[:,[1,14,10,8,0,11,12,3,9,13,4,5,6,8,2]]

all_true_labels = []
all_pred_probs = []

i=1
for i in range(1,16):
    X_train=train_x[:,0:i]
    X_test=test_x[:,0:i]
    y_train=train_y
    y_test=test_y

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    print(roc_auc_score(y_test, y_proba[:,1]))
    all_true_labels.append(y_test)
    all_pred_probs.append(y_proba)

true_labels = convert_to_onehot(all_true_labels)
true_labels = np.transpose(true_labels)
merged_array, second_columns = extract_and_merge_columns(all_pred_probs)
pred_probas = merged_array

np.savetxt('D:/R/kq/true_labels.csv', true_labels, delimiter=',')
np.savetxt('D:/R/kq/pred_probas.csv', pred_probas, delimiter=',')

###################################################################
#autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
with open('D:/R/文章全部数据/kqtrain_6vars.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    kqtrain_6vars = list(reader)
kqtrain_6vars=convert(kqtrain_6vars)[0:682,:]

with open('D:/R/文章全部数据/kqtest_6vars.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    kqtrain_6vars = list(reader)
kqtrain_6vars=convert(kqtrain_6vars)

with open('D:/R/文章全部数据/kqval_6vars.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    kqtest_6vars = list(reader)
kqtest_6vars=convert(kqtest_6vars)

train_data = TabularDataset(kqtrain_6vars)
test_data = TabularDataset(kqtest_6vars)


label = 0

predictor = TabularPredictor(label=label).fit(train_data)



y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()

predictor.evaluate(test_data)

predictor.leaderboard(test_data)

test_proba = predictor.predict_proba(test_data)
roc_auc_score(test_data.loc[:,0], test_proba.loc[:,1])
np.savetxt('D:/R/kq/val_autogluon.csv', test_proba, delimiter=',')
########################################################################################
#病理
train_y, train_x = read_csv_columns('D:/R/文章全部数据/kqtrain_6vars.csv')
test_y, test_x = read_csv_columns('D:/R/文章全部数据/kqtest_oscc.csv')
train_x = convert(train_x)
test_x = convert(test_x)
train_y = convert(train_y)
test_y = convert(test_y)
train_y=[int(x) for x in train_y]
test_y=[int(x) for x in test_y]
train_y=np.array(train_y)
test_y=np.array(test_y)
clf = TabPFNClassifier(ignore_pretraining_limits=True,device='cpu',model_path='D:\\R\\model\\models\\models\\tabpfn-v2.5-classifier-v2.5_default.ckpt')  # Uses TabPFN 2.5 weights, finetuned on real data.
clf.fit(train_x, train_y)
prediction_probabilities = clf.predict_proba(test_x)
np.savetxt('D:/R/pred_oscc.csv', prediction_probabilities, delimiter=',')
