

import cv2
import numpy as np
import argparse

import myFunc as mf
import colorList as cl

import digit_recognize as dr

# 自定义的常量
mask_size = 11
minArea = 30
minHWratio = 1
maxHWratio = 3
RED = 1
BLUE = 2

# 自定义的list
data_list_red = []  # 通过红色筛选后的轮廓
first_data_red = [] # 第一次筛选后的轮廓
second_data1_red = []   # 第二次筛选后的灯柱1
second_data2_red = []   # 第二次筛选后的灯柱2

data_list_blue = [] # 第一次通过蓝色筛选后的轮廓
first_data_blue = []
second_data1_blue = []
second_data2_blue = []

roi = []
paths = []

# TODO: Video 算法

def get_color(frame,color=RED):
    if color == RED:
        print('go in get_red3')
        try:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            color_dict = cl.getColorList()
            mask = cv2.inRange(hsv,color_dict['red3'][0],color_dict['red3'][1])
            cv2.imwrite('red_mask.jpg',mask)
            blurred = cv2.medianBlur(mask,mask_size)
            cv2.imwrite('red_blurred.jpg',blurred)
            binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary,None,iterations=2)
            cv2.imwrite("red_binary.jpg",binary)
            return (hsv,gray,binary)
        except:
            print("there is no red elements")
            return (None,None,None)
    elif color == BLUE:
        try:
            print('go in get_blue2')
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            color_dict = cl.getColorList()
            mask = cv2.inRange(hsv,color_dict['blue2'][0],color_dict['blue2'][1])
            cv2.imwrite('blue_mask.jpg',mask)
            blurred = cv2.medianBlur(mask,mask_size)
            cv2.imwrite('blue_blurred.jpg',blurred)
            binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary,None,iterations=2)
            cv2.imwrite("blue_binary.jpg",binary)
            return (hsv,gray,binary)
        except:
            print("there is no blue elements")
            return (None,None,None)
    else:
        print("Only Accept Red & Blue")


def detect_armor_red_image(frame,gray,binary):
    if gray is not None and binary is not None:
        img1 = frame.copy() #备份1
        img2 = frame.copy()
        (cnts, hierachy) = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        amount = len(cnts)
        print("contours amount:",amount)
        if amount > 0:
            print("-----found------")
            for i, cnt in enumerate(cnts):
                data_dict = dict()
                # print("contour",contour)
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
                cv2.drawContours(img1,[box],0,(0,0,255),2)  # 第一个参数是InputOutput
                
                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[1][0]
                y2 = box[1][1]
                x3 = box[2][0]
                y3 = box[2][1]
                x4 = box[3][0]
                y4 = box[3][1]

                data_dict["area"] = area
                data_dict["rx"] = rx
                data_dict["ry"] = ry
                data_dict["rw"] = rh
                data_dict["rh"] = rw        # 注意 我这里把rw和rh互换了， 主要是为了对应物理意义
                data_dict["zeta"] = zeta
                data_dict["x1"] = x1
                data_dict["y1"] = y1
                data_dict["x2"] = x2
                data_dict["y2"] = y2
                data_dict["x3"] = x3
                data_dict["y3"] = y3
                data_dict["x4"] = x4
                data_dict["y4"] = y4
                data_dict["four_points"] = box
                data_list_red.append(data_dict)
            print("data_list_red",len(data_list_red))
            # print("data_dict",len(data_dict))

            mf.cv_show("preprocessing",img1)

            for i in range(len(data_list_red)):
                # 第一次筛选，通过长宽比把可能值放入first_data列表中
                data_rh = data_list_red[i].get("rh", 0)
                data_rw = data_list_red[i].get("rw", 0)
                data_area = data_list_red[i].get("area", 0) 

                # 高 > 宽， 面积不能太小
                if (float(data_rh / data_rw) >= minHWratio) \
                        and (float(data_rh / data_rw) <= maxHWratio) \
                        and data_area >= minArea:
                    first_data_red.append(data_list_red[i])
                else:
                    pass
                
            print("first_data_red:",len(first_data_red))
            # print("first_data_0:",first_data[0])

            # 检测筛选的第一波数值 
            for i in range(len(first_data_red)):
                four_points = first_data_red[i].get("four_points",0)
                cv2.drawContours(img2,[four_points],0,(0,255,0),2)
            
            cv2.imwrite("first-filter-red.jpg",img2)


            for i in range(len(first_data_red)):

                c = i + 1
                while c < len(first_data_red):
                    data_ryi = float(first_data_red[i].get("ry", 0))    # 0表示如果指定键不存在时，返回值为0
                    data_ryc = float(first_data_red[c].get("ry", 0))
                    data_rhi = float(first_data_red[i].get("rh", 0))
                    data_rhc = float(first_data_red[c].get("rh", 0))
                    data_rxi = float(first_data_red[i].get("rx", 0))
                    data_rxc = float(first_data_red[c].get("rx", 0))
                    four_points_i = first_data_red[i].get("four_points",0)
                    four_points_c = first_data_red[c].get("four_points",0)

                    # 应该是对每两个灯条进行识别配比，来确定是不是装甲板上相邻的灯条 (可修改参数)
                    h_distance = 0.2 * max(data_rhi, data_rhc)
                    x_distance = 4 * ((data_rhi + data_rhc) / 2)
                    y_distance = 2 * ((data_rhi + data_rhc) / 2)

                    if (abs(data_ryi - data_ryc) <= y_distance) \
                            and (abs(data_rhi - data_rhc) <= h_distance) \
                            and (abs(data_rxi - data_rxc) <= x_distance):

                        # 做两两匹配，得到两个相邻的灯
                        second_data1_red.append(first_data_red[i])
                        second_data2_red.append(first_data_red[c])

                        cv2.drawContours(frame,[four_points_i],0,(0,255,0),2)
                        cv2.drawContours(frame,[four_points_c],0,(0,255,0),2)
                    c = c + 1

            print("second_data1_red ",len(second_data1_red))
            print("second_data1_red",second_data1_red)
            print("second_data2_red ",len(second_data1_red))
            print("second_data2_red ",second_data2_red)
            # cv2.drawContours(frame,[second_data1[0].get("x1",0)],0,(0,255,0),2)
            # cv2.drawContours(frame,[second_data2[0].get("x1",0)],0,(0,255,0),2)
            # cv2.rectangle(frame,(second_data1[0].get("x1",0),second_data1[0].get("y1",0)),(second_data1[0].get("x3",0),second_data1[0].get("y3",0)),(0,255,100),2)
            # cv2.rectangle(frame,(second_data2[0].get("x1",0),second_data2[0].get("y1",0)),(second_data2[0].get("x3",0),second_data2[0].get("y3",0)),(0,255,100),2)
            cv2.imwrite("second-filter-red.jpg",frame)


            if len(second_data1_red):
                for i in range(len(second_data1_red)):
                    print("i:   ",i)
                    gray_copy = gray.copy() 
                    
                    rectangle_x1 = int(second_data1_red[i]["x1"])   # 左上
                    rectangle_y1 = int(second_data1_red[i]["y1"])
                    rectangle_x2 = int(second_data2_red[i]["x3"])   # 右下
                    rectangle_y2 = int(second_data2_red[i]["y3"])

                    # if abs(rectangle_y1 - rectangle_y2) <=  (abs(rectangle_x1 - rectangle_x2)): 这里要注意！
                    if abs(rectangle_y1 - rectangle_y2) <= (6 / 2) *(abs(rectangle_x1 - rectangle_x2)):
                        
                        
                        # Point 1的点
                        point1_1x = second_data1_red[i]["x1"]
                        point1_1y = second_data1_red[i]["y1"]
                        point1_2x = second_data1_red[i]["x2"]
                        point1_2y = second_data1_red[i]["y2"]
                        point1_3x = second_data1_red[i]["x3"]
                        point1_3y = second_data1_red[i]["y3"]
                        point1_4x = second_data1_red[i]["x4"]
                        point1_4y = second_data1_red[i]["y4"]
                        point1_rh = second_data1_red[i]['rh']

                        # Point 2的点
                        point2_1x = second_data2_red[i]["x1"]
                        point2_1y = second_data2_red[i]["y1"]
                        point2_2x = second_data2_red[i]["x2"]
                        point2_2y = second_data2_red[i]["y2"]
                        point2_3x = second_data2_red[i]["x3"]
                        point2_3y = second_data2_red[i]["y3"]
                        point2_4x = second_data2_red[i]["x4"]
                        point2_4y = second_data2_red[i]["y4"]
                        point2_rh = second_data2_red[i]['rh']

                        # 两灯柱之间画长方形 -> point1 在右侧， point2 在左侧
                        if point1_1x > point2_1x:
                            pass

                        else:
                            point1_1x, point2_1x = point2_1x, point1_1x
                            point1_2x, point2_2x = point2_2x, point1_2x
                            point1_3x, point2_3x = point2_3x, point1_3x
                            point1_4x, point2_4x = point2_4x, point1_4x

                            point1_1y, point2_1y = point2_1y, point1_1y
                            point1_2y, point2_2y = point2_2y, point1_2y
                            point1_3y, point2_3y = point2_3y, point1_3y
                            point1_4y, point2_4y = point2_4y, point1_4y

                        # 数字框架ROI可以改这里 (可修改参数)
                        left_x = int(point2_2x)
                        left_y = int(point2_2y - point2_rh/2)
                        right_x = int(point1_4x)
                        right_y = int(point1_4y + point2_rh/2)
                        width = abs(right_x - left_x)
                        height = abs(right_y - left_y)
                        num_roi = (left_x,left_y,right_x,right_y,width,height)
                        # print()
                        # print(left_x)
                        # print(left_y)
                        # print(width)
                        # print(height) 
                        
                        cv2.rectangle(frame, (left_x, left_y), (right_x, right_y), (255, 255, 0), 2)
                        
                        number_img = gray_copy[left_y:left_y+height,left_x:left_x+width] 

                        output = dr.digit_detect(number_img)
                        print(output)

                        cv2.putText(frame, "red: "+str(output), (left_x, left_y-5), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,
                            color=(255, 255, 255), thickness=1)
                        
                cv2.imwrite("target-red.jpg",frame)

            else:
                print("---red not find---")
                pass
    else:
        print("---ERROR---")

def detect_armor_blue_image(frame,gray,binary):
    if gray is not None and binary is not None:
        img1 = frame.copy() #备份1
        img2 = frame.copy()
        (cnts, hierachy) = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        amount = len(cnts)
        print("contours amount:",amount)
        if amount > 0:
            print("-----found------")
            for i, cnt in enumerate(cnts):
                data_dict = dict()
                # print("contour",contour)
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
                cv2.drawContours(img1,[box],0,(0,0,255),2)  # 第一个参数是InputOutput
                
                x1 = box[0][0]
                y1 = box[0][1]
                x2 = box[1][0]
                y2 = box[1][1]
                x3 = box[2][0]
                y3 = box[2][1]
                x4 = box[3][0]
                y4 = box[3][1]

                data_dict["area"] = area
                data_dict["rx"] = rx
                data_dict["ry"] = ry
                data_dict["rw"] = rh
                data_dict["rh"] = rw        # 注意 我这里把rw和rh互换了， 主要是为了对应物理意义
                data_dict["zeta"] = zeta
                data_dict["x1"] = x1
                data_dict["y1"] = y1
                data_dict["x2"] = x2
                data_dict["y2"] = y2
                data_dict["x3"] = x3
                data_dict["y3"] = y3
                data_dict["x4"] = x4
                data_dict["y4"] = y4
                data_dict["four_points"] = box
                data_list_blue.append(data_dict)
            print("data_list_blue",len(data_list_blue))
            # print("data_dict",len(data_dict))

            mf.cv_show("preprocessing",img1)

            for i in range(len(data_list_blue)):
                # 第一次筛选，通过长宽比把可能值放入first_data列表中
                data_rh = data_list_blue[i].get("rh", 0)
                data_rw = data_list_blue[i].get("rw", 0)
                data_area = data_list_blue[i].get("area", 0) 

                # 高 > 宽， 面积不能太小
                if (float(data_rh / data_rw) >= minHWratio) \
                        and (float(data_rh / data_rw) <= maxHWratio) \
                        and data_area >= minArea:
                    first_data_blue.append(data_list_blue[i])
                else:
                    pass
                
            print("first_data_blue:",len(first_data_blue))
            # print("first_data_0:",first_data[0])

            # 检测筛选的第一波数值 
            for i in range(len(first_data_blue)):
                four_points = first_data_blue[i].get("four_points",0)
                cv2.drawContours(img2,[four_points],0,(0,255,0),2)
            
            cv2.imwrite("first-filter-blue.jpg",img2)


            for i in range(len(first_data_blue)):

                c = i + 1
                while c < len(first_data_blue):
                    data_ryi = float(first_data_blue[i].get("ry", 0))    # 0表示如果指定键不存在时，返回值为0
                    data_ryc = float(first_data_blue[c].get("ry", 0))
                    data_rhi = float(first_data_blue[i].get("rh", 0))
                    data_rhc = float(first_data_blue[c].get("rh", 0))
                    data_rxi = float(first_data_blue[i].get("rx", 0))
                    data_rxc = float(first_data_blue[c].get("rx", 0))
                    four_points_i = first_data_blue[i].get("four_points",0)
                    four_points_c = first_data_blue[c].get("four_points",0)

                    # 应该是对每两个灯条进行识别配比，来确定是不是装甲板上相邻的灯条 (可修改参数)
                    h_distance = 0.2 * max(data_rhi, data_rhc)
                    x_distance = 4 * ((data_rhi + data_rhc) / 2)
                    y_distance = 2 * ((data_rhi + data_rhc) / 2)

                    if (abs(data_ryi - data_ryc) <= y_distance) \
                            and (abs(data_rhi - data_rhc) <= h_distance) \
                            and (abs(data_rxi - data_rxc) <= x_distance):

                        # 做两两匹配，得到两个相邻的灯
                        second_data1_blue.append(first_data_blue[i])
                        second_data2_blue.append(first_data_blue[c])

                        cv2.drawContours(frame,[four_points_i],0,(0,255,0),2)
                        cv2.drawContours(frame,[four_points_c],0,(0,255,0),2)
                    c = c + 1

            print("second_data1_blue ",len(second_data1_blue))
            print("second_data1_blue",second_data1_blue)
            print("second_data2_blue ",len(second_data1_blue))
            print("second_data2_blue ",second_data2_blue)
            # cv2.drawContours(frame,[second_data1[0].get("x1",0)],0,(0,255,0),2)
            # cv2.drawContours(frame,[second_data2[0].get("x1",0)],0,(0,255,0),2)
            # cv2.rectangle(frame,(second_data1[0].get("x1",0),second_data1[0].get("y1",0)),(second_data1[0].get("x3",0),second_data1[0].get("y3",0)),(0,255,100),2)
            # cv2.rectangle(frame,(second_data2[0].get("x1",0),second_data2[0].get("y1",0)),(second_data2[0].get("x3",0),second_data2[0].get("y3",0)),(0,255,100),2)
            cv2.imwrite("second-filter-blue.jpg",frame)


            if len(second_data1_blue):
                for i in range(len(second_data1_blue)):
                    print("i:   ",i)
                    gray_copy = gray.copy() 
                    
                    rectangle_x1 = int(second_data1_blue[i]["x1"])   # 左上
                    rectangle_y1 = int(second_data1_blue[i]["y1"])
                    rectangle_x2 = int(second_data2_blue[i]["x3"])   # 右下
                    rectangle_y2 = int(second_data2_blue[i]["y3"])

                    # if abs(rectangle_y1 - rectangle_y2) <=  (abs(rectangle_x1 - rectangle_x2)): 这里要注意！
                    if abs(rectangle_y1 - rectangle_y2) <= (6 / 2) *(abs(rectangle_x1 - rectangle_x2)):
                        
                        
                        # Point 1的点
                        point1_1x = second_data1_blue[i]["x1"]
                        point1_1y = second_data1_blue[i]["y1"]
                        point1_2x = second_data1_blue[i]["x2"]
                        point1_2y = second_data1_blue[i]["y2"]
                        point1_3x = second_data1_blue[i]["x3"]
                        point1_3y = second_data1_blue[i]["y3"]
                        point1_4x = second_data1_blue[i]["x4"]
                        point1_4y = second_data1_blue[i]["y4"]
                        point1_rh = second_data1_blue[i]['rh']

                        # Point 2的点
                        point2_1x = second_data2_blue[i]["x1"]
                        point2_1y = second_data2_blue[i]["y1"]
                        point2_2x = second_data2_blue[i]["x2"]
                        point2_2y = second_data2_blue[i]["y2"]
                        point2_3x = second_data2_blue[i]["x3"]
                        point2_3y = second_data2_blue[i]["y3"]
                        point2_4x = second_data2_blue[i]["x4"]
                        point2_4y = second_data2_blue[i]["y4"]
                        point2_rh = second_data2_blue[i]['rh']

                        # 两灯柱之间画长方形 -> point1 在右侧， point2 在左侧
                        if point1_1x > point2_1x:
                            pass

                        else:
                            point1_1x, point2_1x = point2_1x, point1_1x
                            point1_2x, point2_2x = point2_2x, point1_2x
                            point1_3x, point2_3x = point2_3x, point1_3x
                            point1_4x, point2_4x = point2_4x, point1_4x

                            point1_1y, point2_1y = point2_1y, point1_1y
                            point1_2y, point2_2y = point2_2y, point1_2y
                            point1_3y, point2_3y = point2_3y, point1_3y
                            point1_4y, point2_4y = point2_4y, point1_4y

                        # 数字框架ROI可以改这里 (可修改参数)
                        left_x = int(point2_2x)
                        left_y = int(point2_2y - point2_rh/2)
                        right_x = int(point1_4x)
                        right_y = int(point1_4y + point2_rh/2)
                        width = abs(right_x - left_x)
                        height = abs(right_y - left_y)
                        num_roi = (left_x,left_y,right_x,right_y,width,height)
                        # print()
                        # print(left_x)
                        # print(left_y)
                        # print(width)
                        # print(height) 
                        
                        cv2.rectangle(frame, (left_x, left_y), (right_x, right_y), (255, 255, 0), 2)
                        
                        number_img = gray_copy[left_y:left_y+height,left_x:left_x+width] 

                        output = dr.digit_detect(number_img)
                        print(output)

                        cv2.putText(frame, "blue: "+str(output), (left_x, left_y-5), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,
                            color=(255, 255, 255), thickness=1)
                        
                cv2.imwrite("target-blue.jpg",frame)

            else:
                print("---Blue not found---")
                pass
    else: 
        print("---ERROR---")

if __name__ == '__main__':
    filename = "images/img2.jpg"
    frame = cv2.imread(filename)
    (hsv,gray,binary) = get_color(frame,1)
    detect_armor_red_image(frame,gray,binary)
    (hsv,gray,binary) = get_color(frame,2)
    detect_armor_blue_image(frame,gray,binary)