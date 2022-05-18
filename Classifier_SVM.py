import numpy as np
from sklearn.svm import LinearSVC
import joblib
from Dataset_process import *

def create_train_dataset():
    #加载数据集，数据集结构为[6类x每类样本数x[10x2]]，每个样本记录十个关健身体点
    data = np.load("posedata.npy",allow_pickle=True)
    num_samples_0 = len(data[0])
    num_samples_1 = len(data[1])
    num_samples_2 = len(data[2])
    num_samples_3 = len(data[3])
    num_samples_4 = len(data[4])
    num_samples_5 = len(data[5])
    all_samples = [num_samples_0,num_samples_1,num_samples_2,num_samples_3,num_samples_4,num_samples_5]


    #标签列表
    label = [0]*num_samples_0 + [1]*num_samples_1 + [2]*num_samples_2 + [3]*num_samples_3 + [4]*num_samples_4 + [5]*num_samples_5
    label = np.asarray(label)



    #训练集格式[len(label) x [20]]
    dataset = []
    for i in range(6):
        dataset += list(np.asarray(data[i]).reshape((all_samples[i],20)))
    dataset = np.asarray(dataset)

    return label, dataset

#分类器
def Classifier(dataset,label):
    clf = LinearSVC()
    x = dataset
    y = label
    clf.fit(x,y)
    joblib.dump(clf, "quanhuang_pose_model.m")

#预测,输入数据格式[[20,]]
def predict(data):
    model = joblib.load("quanhuang_pose_model.m")
    res = model.decision_function(data)

    return res

if __name__ == "__main__":
    #加载训练集，训练模型
    label, dataset = create_train_dataset()
    Classifier(dataset,label)

    '''
    #测试一下模型
    test_data = body_data_generate("raw_pose_4_18.npy")
    if -1 not in test_data:
        res = predict([test_data.reshape((20,))])
        #大于0的index对应为判定类
        print(res)
    '''
