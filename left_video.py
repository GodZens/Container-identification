import time
import datetime
import cv2
import numpy as np
from PIL import Image
import argparse
from yolo import YOLO as yolov4_1
from yolo_2 import YOLO as yolov4_2
from single_frame_process import get_code_and_draw
import csv
import os

def get_files(files_dir):
    files_list = []      
    for file in os.listdir(files_dir):
        if(file=='test.txt'):
            continue
        files_list.append(file)
    return  files_list

def argument_parser(video):
    """ Handle command line arguments """
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', default=f'{video}', help='path to input video')
    args = vars(ap.parse_args(args=[]))
    return args

def resize_box(frame,left,right,top,bottom):
    if(left<0):
        left = 0
    if(right>np.size(frame,1)):
        right = np.size(frame,1)
    if(top<0):
        top = 0
    if(bottom>np.size(frame,0)):
        bottom = np.size(frame,0)
    return left,right,top,bottom

def crop_img(frame,predicted_class,boxs):
    crop_frame=[]
    re_box = []
    n_obj = len(predicted_class)
    for i in range(n_obj):
        #擴大yolov4_1_box所框範圍
        box = boxs[i]
        top, left, bottom, right = box
        extra_pix = 5  # Get a bit bigger crop to make sure we cover everything
        left = int(left-extra_pix)
        right = int(right+extra_pix)
        top = int(top-extra_pix)
        bottom = int(bottom+extra_pix)
        left,right,top,bottom = resize_box(frame,left,right,top,bottom)
        crop_frame = frame[top:bottom,left:right]
        re_box = [top, left, bottom, right]
    return crop_frame,re_box

# def main(args):
def main():
    start = time.time()
    #載入模型
    yolo_1 = yolov4_1()
    yolo_2 = yolov4_2()
    #讀取影片
    capture = cv2.VideoCapture('rtsp://admin:dh123456@192.168.1.102')
    # capture = cv2.VideoCapture(args['video'])

    ref, frame = capture.read()
    if not ref:
        raise ValueError("路徑錯誤")

    fps = 0.0
    count = 0
    temp=[]
    flag=True
    while(True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        count+=1
        #抓取影片每兩帧的圖片
        if count % 2 == 0:
            codes=[]
            # 格式转变，BGRtoRGB
            frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame,predicted_class,boxs,score = np.array(yolo_1.detect_image(frame))

            # RGBtoBGR
            frame = np.asarray(frame)
            r, g, b = cv2.split(frame)
            frame=cv2.merge([b, g, r])
            
            crop_frame,re_box = crop_img(frame,predicted_class,boxs)

            if(len(crop_frame)!=0):
                # 转变成Image
                crop_array_img = Image.fromarray(np.uint8(crop_frame))
                frame_2,predicted_class_2,boxs_2,score_2 = np.array(yolo_2.detect_image(crop_array_img))
                print("predicted_class_2:",predicted_class_2)
                if(len(predicted_class_2)!=0):
                    frame, codes = get_code_and_draw(frame,crop_frame,re_box,predicted_class_2,debug=False)
                if codes!=[''] and codes!=[] and len(codes)==1:   
                    temp.append(codes)
            else:
                interval_start = time.time()
            
            if(len(temp) == 20):
                result=max(temp,key=temp.count)
                if(flag):
                    session = result
                    localtime = time.localtime()
                    date = time.strftime("%Y-%m-%d %H:%M:%S",localtime)
                    now = datetime.datetime.now()
                    everyday = "Z:\\38.DLV_Python02_to_OA\\"
                    filename = everyday+\
                        session[0]+"_"+now.strftime("%Y%m%d%H%M%S")+"_L"+".csv"
                    with open(filename, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        with open(filename, "r", newline="") as f:
                            reader = csv.reader(f)
                            if not [row for row in reader]:
                                writer.writerow(['Lane','Time','Container Number','Check Digit'])
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()
                            else:
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()
                    records = "C:\\Users\\python02\\Desktop\\records2\\container.csv"
                    with open(records, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        with open(records, "r", newline="") as f:
                            reader = csv.reader(f)
                            if not [row for row in reader]:
                                writer.writerow(['Lane','Time','Container Number','Check Digit'])
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()
                            else:
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()                    
                    flag = False
                if(session != result):
                    session=result
                    localtime = time.localtime()
                    date = time.strftime("%Y-%m-%d %H:%M:%S",localtime)
                    now = datetime.datetime.now()
                    everyday = "Z:\\38.DLV_Python02_to_OA\\"
                    filename = everyday+\
                        session[0]+"_"+now.strftime("%Y%m%d%H%M%S")+"_L"+".csv"
                    with open(filename, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        with open(filename, "r", newline="") as f:
                            reader = csv.reader(f)
                            if not [row for row in reader]:
                                writer.writerow(['Lane','Time','Container Number','Check Digit'])
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()
                            else:
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()
                    records = "C:\\Users\\python02\\Desktop\\records2\\container.csv"
                    with open(records, 'a+', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        with open(records, "r", newline="") as f:
                            reader = csv.reader(f)
                            if not [row for row in reader]:
                                writer.writerow(['Lane','Time','Container Number','Check Digit'])
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close()
                            else:
                                writer.writerow(['左車道',date,session[0],session[0][10]])
                                csvfile.close()
                                f.close() 
                temp=[]
            else:
                interval_end=time.time()
                interval_time = interval_start-interval_end
                if(int(interval_time) > 30):
                    print("interval_time",interval_time)
                    temp=[]


            #將fps寫在影片上
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             cv2.namedWindow("video2",cv2.WINDOW_NORMAL)
#             cv2.setWindowProperty("video2",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("video2",frame)
            c= cv2.waitKey(1) & 0xff 
            if c==27:
                capture.release()
                break
    print("Video Detection Done!")        
    capture.release()
    cv2.destroyAllWindows()            


if __name__ == "__main__":
    main()
#       video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
#                           想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
#     files_dir = 'C:\\Users\\idsl\\Desktop\\展示錄影影片\\錄影'#資料夾路徑
#     test_list = get_files(files_dir) 
#     for i in range(len(test_list)):
#         video_path      = f"{files_dir}\\{test_list[i]}"
#         arguments = argument_parser(video_path)
#         main(arguments)     

#     files_dir = 'E:\\白色貨櫃\\影片\\總測試資料'#資料夾路徑
#     test_list = get_files(files_dir) 
#     print(test_list)
#     for i in range(len(test_list)):
#         video_path      = f"{files_dir}\\{test_list[i]}"
#         arguments = argument_parser(video_path)
#         main(arguments)
#     video_path      = f"C:\\Users\\idsl\\Desktop\\未辨識出來\\YMMU6159867.mp4"
#     arguments = argument_parser(video_path)
#     main(arguments)
