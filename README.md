# Change Log
5.24: 主程序进行了线程优化，识别控制与动作展示界面分开，运行更加流畅
5.24：The main program has been optimized to run more smoothly.

# Mediapipe_App
Some Applications of Mediapipe

本仓库中会上传一些基于MediaPipe开发的小程序

# quanhaung_run.py
A programm using camera to control keyboard and play games.

通过电脑摄像头识别身体姿势来玩拳皇游戏

## How to use
1. 运行rawdata_generate.py，拍一些用于控制对应按键的动作照片，注意需要修改对应的图片保存名称，命名格式："raw_pose_typenumber_imagenumber",
如"raw_pose_1_0",表示第一类姿势的第一张图片

（run the "rawdata_generate.py" to take some photos with different poses as the raw-data to train a classifier. You should change the name of image
according to the pose-class. Name-Format: "raw_pose_typenumber_imagenumber", excample "raw_pose_1_0" means the first image of the first pose-class.）

2. 运行Dataset_process.py，从拍摄图片中提取用于表现身体姿态的关键点并生成训练集，posedata.npy

(run "Dataset_process.py" to extract the some important body-points from images and create the training-datasets "posedata.npy")

3. 运行Classifier_SVM.py来训练并保存模型 "quanhuang_pose_model.m"

(run "Classifier_SVM.py to train and save the Classifier-Model "quanhuang_pose_model.m")

4. 修改Actions.py中各个姿势类对应的按键（也可以调整一些其他参数以优化游戏体验）

(Modify the keys in "Actions.py" corresponding to your pose-class-number. You can also change some other parameters.)

5. 运行主程序quanhuang_run.py,开启摄像头，玩拳皇~

（Now run the main program "quanhuang_run.py" to open camera, then start a game to play）
