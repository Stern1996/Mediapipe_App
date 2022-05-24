import cv2
import mediapipe as mp
import numpy as np
from Actions import *
import pyautogui
from Dataset_process import *
from Classifier_SVM import *
import time
import threading
from Actions import *
import pickle

# 动作控制线程
def action_predict():
    while True:
        time.sleep(0.2)
        with open("posedata.txt","rb") as f:
            try:
                pose_data = pickle.load(f)
                pose_data = landmark_parse(pose_data)
                #判断当前图像是否包含所需检测关键点数据,然后传入按键判断中
                if -1 not in pose_data:
                    info = actions(pose_data)
                else:
                    info = "No Pose"
            except EOFError:
                pass

def run():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose


    #测试用的时间变量
    start_time = time.time()

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            # 将landmarks保存入临时文件供判断线程使用
            with open("posedata.txt",'wb') as f:
                pickle.dump(results.pose_landmarks,f)
            # pose_data = results.pose_landmarks
            # pose_data = landmark_parse(pose_data)
            # #判断当前图像是否包含所需检测关键点数据,然后传入按键判断中
            # if -1 not in pose_data and time.time()-start_time > 0.2:
            #     info = actions(pose_data)
            #     start_time = time.time()
            # else:
            #     info = "No Pose"
            #
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.flip(image,1)
            #image = show_result_image(image,info)

        else:
            image = cv2.flip(image,1)


        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()

if __name__ == "__main__":
    t = threading.Thread(target=action_predict,daemon=True)
    t.start()
    run()
