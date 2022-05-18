#一些对landmark数据格式化和处理的方法
import numpy as np
import mediapipe
import os

#处理收集的原始npy数据，输出对应的四肢各关节点数据信息，结构[10，2],以面部初始0点为基准
def body_data_generate(filepath):
    data = np.load(filepath, allow_pickle=True)
    points = []
    # 身体各点编号列表，取部分关键点
    body_list = list(range(11, 17)) + list(range(23,27))
    #基准点
    base = data[0].landmark[0]
    #遍历所有点，判断visibility后建立body_points列表
    for idx, landmark in enumerate(data[0].landmark):
        if (idx in body_list and landmark.visibility < 0.5):
            points.append(-1)
        elif (idx in body_list and landmark.visibility > 0.5):
            points.append([landmark.x-base.x, landmark.y-base.y])

    return np.asarray(points)

#直接对传入landmark类型数据进行处理
def landmark_parse(landmark_data):
    data = landmark_data
    points = []
    # 身体各点编号列表，取部分关键点
    body_list = list(range(11, 17)) + list(range(23, 27))
    # 基准点
    base = data.landmark[0]
    # 遍历所有点，判断visibility后建立body_points列表
    for idx, landmark in enumerate(data.landmark):
        if (idx in body_list and landmark.visibility < 0.5):
            points.append(-1)
        elif (idx in body_list and landmark.visibility > 0.5):
            points.append([landmark.x - base.x, landmark.y - base.y])

    return np.asarray(points,dtype=object)



if __name__ == "__main__":
    filenames = os.listdir("./images")
    #index从前到后依次对应上下左右拳脚，故type最大为5,共六个动作
    posedata = [[],[],[],[],[],[]]
    for file in filenames:
        if "npy" in file:
            filepath = "./images/"+file
            type = int(file.split("_")[2])
            data = body_data_generate(filepath)
            #剔除未识别到全部身体坐标点的图片
            if -1 not in data:
                posedata[type].append(data)
    posedata = np.asarray(posedata)
    #建立用于训练分类器的数据集
    np.save("posedata.npy",posedata)