model_data:主要是用來存放yolov7的訓練權重檔
net:主要是yolo系列的骨幹架構、neck和head
src:用來進行資料前處理的地方
utils:工具
left_video.py:左車道影像的辨識貨櫃號碼程式
right_video.py:右車道影像的辨識貨櫃號碼程式
single_frame_process.py:辨識單張圖片的程式
yolo.py:yolo的程式碼
yolo_2.py:第二個yolo的程式碼
本貨櫃辨識是用yolov7做的，其內部有加入過ssh、cbma等模塊來幫助機器更能增測到小物件
yolo部分可以參考bubbliiiing的github https://github.com/bubbliiiing/yolov4-pytorch
我是使用他所創建的yolov4和yolov7來訓練模型的
 
