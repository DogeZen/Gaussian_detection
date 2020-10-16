import time

import cv2
import numpy as np



def add_new_cam(rtsp_path, points):
    # 创建模型
    mog = cv2.createBackgroundSubtractorMOG2()  # 定义高斯混合模型对象 mog
    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    knn = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    # 绘制蒙版
    cap = cv2.VideoCapture(rtsp_path)
    ret, frame = cap.read()
    mask = np.zeros(frame.shape, np.uint8)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))


    # 初始化计时器用于判断时间
    time_now = time.time()

    cv2.imshow("mask", mask)
    while 1:
        ret, frame = cap.read()
        frame_to_save = frame.copy()
        frame_to_show = frame.copy()
        frame = cv2.bitwise_and(frame, mask)

        # 混合高斯模型
        fgmask = mog.apply(frame)  # 使用前面定义的高斯混合模型对象 mog 当前帧的运动目标检测，返回二值图像
        gray_frame = fgmask.copy()
        kernel = np.ones((5, 5), np.uint8)
        gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)
        # 返回值： contours，轮廓的坐标。 hierarchy，各个框之间父子关系，不常用。
        contours, hierarchy = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制每一个轮廓框到原始图像 frame 中
        for contour in contours:
            if cv2.contourArea(contour) < 1500:  # 计算候选框的面积，如果小于1500，跳过当前候选框
                continue
            (x, y, w, h) = cv2.boundingRect(contour)  # 根据轮廓，得到当前最佳矩形框
            cv2.rectangle(frame_to_show, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 将该矩形框画在当前帧 frame 上

            # 根据时间间隔保存图片
            interval = time.time() - time_now
            time_now = time.time()
            if interval > 5:
                # 保存图片
                cv2.imwrite(filename="image/" + str(time_now) + ".jpg", img=frame_to_save)

        cv2.imshow("gray", gray_frame)
        cv2.imshow("contours", frame_to_show)  # 显示当前帧
        cv2.waitKey(30)


rtsp = "rtsp://admin:Buchou123@192.168.1.64:554/h264/ch1/main/av_stream"
points = np.array([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
add_new_cam(rtsp, points)
