import time

from Classifier_SVM import *
import pyautogui
import cv2



def actions(posedata):
    pose_data = posedata.reshape((1,-1))
    res = predict(pose_data)
    info = "No Pose"
    #根据预测结果做出相应动作
    if res.max() < 0:
        pyautogui.keyUp('w')
        pyautogui.keyUp('s')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        pyautogui.keyUp('j')
        pyautogui.keyUp('k')
        info = "No Pose！"
    #设置个置信阈值，不然会很容易乱输出
    #判断当前帧的动作与上个动作是否相同
    elif res.max() > 0.4 and res.argmax() == 0:
        pyautogui.keyDown('w')
        info = "Jump"
    elif res.max() > 0.4 and res.argmax() == 1:
        pyautogui.keyDown('s')
        info = "Down"
    elif res.max() > 0.4 and res.argmax() == 2:
        pyautogui.keyDown('a')
        info = "Move Left"
    elif res.max() > 0.4 and res.argmax() == 3:
        pyautogui.keyDown('d')
        info = "Move Right"
    elif res.max() > 0.1 and res.argmax() == 4:
        pyautogui.keyDown('j')
        info = "Punch"
    elif res.max() > 0.4 and res.argmax() == 5:
        pyautogui.keyDown('k')
        info = "Kick"

    return info

def show_result_image(img,info):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    #输出监测信息到图像上
    out_img = cv2.putText(img,info, org, font,
               fontScale, color, thickness, cv2.LINE_AA)

    return out_img