

import cv2
import numpy as np
from imutils.perspective import four_point_transform


def digit_detect(roi):
    '''
        该方法是用于匹配ROI和模板找到对应数值
        roi: 需要匹配的数字区域
        return: 返还改区域的数字
    '''
    try:
        resize = (24,40)    # 可修改
        scores = [] # scores 用来查询匹配度最高的值
        # print(roi)    roi调用过来的时候本身就是灰色图
        # target.astype(np.uint8)
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        # cv_show("blurred",blurred)
        ret,target_binary = cv2.threshold(blurred,50,255,cv2.THRESH_BINARY)
        # cv_show("Binary",target_binary)
        cnts, _ = cv2.findContours(target_binary.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnt = cnts[0]   # 最大的一定是
        
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt) # consider the rotational rectangle
        # 函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
        #分别对应于返回值：(rect[0][0],  rect[0][1]),  (rect[1][0],  rect[1][1]),  rect[2] 
        rx, ry = rect[0]
        rw, rh = rect[1]
        zeta = rect[2]

        coor = cv2.boxPoints(rect)    #获取最小外接矩形的四个顶点 左上，右上，右下，左下
        box = np.int0(coor)            # box 包含四个顶点，用int0是把他们都整数化
        # print(rect) # (951.5, 953.5), (7.0, 11.0), 90.0) 格式

        cv2.drawContours(roi,[box],0,(0,0,255),2)  # 第一个参数是InputOutput
        # cv_show("target-contour",target_copy)

        transformed = four_point_transform(target_binary, box)
        # cv_show("4-points-trans",transformed)


        # 得到了要匹配的目标
        target = cv2.resize(transformed,resize)
        
        # cv_show("final-target",target)
        cv2.imwrite("final-target.jpg",target)
        # print(target.dtype)
        # print(target.ndim)

        # 现在需要的是匹配的模板

        for j in range(8):
            num = j+1
            
            template = cv2.imread("Template\\{}.jpg".format(num),0)     #一定要转成灰度图！让ndim = 2, 这样才能够用matchTemplate
            # template.astype(np.uint8)
            
            # print('Original Dimensions : ',templates.shape)
            
            # resize 成统一格式 （24x40)
            ret,thresh1 = cv2.threshold(template,127,255,cv2.THRESH_BINARY)
            
            template = cv2.resize(thresh1, resize)
            
            # cv_show('template',template)

            # print(template.dtype)
            # print(template.ndim)
            # print('Resized Dimensions : ',resized.shape)

            result = cv2.matchTemplate(target,template,cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            print("num: {0}, score: {1}".format(num,score))
            scores.append(score)
        
        # 得到最合适的数字
        output_index = (np.argmax(scores))   # 这里得到的是在list中的最大值的index
        output = output_index + 1
        print("output:",output)
        return output

    except FileNotFoundError:
        print("File is not found.")
    except PermissionError:
        print("You don't have permission to access this file.")

