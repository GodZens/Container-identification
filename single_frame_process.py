import cv2
from src import ocr
import argparse
from src.code_region_detector import build_model, detect
from src.code_image_cleaner_final import process_image_for_ocr
from src.utils import display_image_cv2, resize_to_suitable
import numpy as np

def argument_parser():
    """ Handle command line arguments """
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', default='demo_input.jpg', help='path to input image')
    ap.add_argument('-c', '--config', default='yolov4.cfg', help='path to yolo config file')
    ap.add_argument('-w', '--weights', default='yolov4.weights', help='path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', default='yolov4.txt', help='path to text file containing class names')
    args = vars(ap.parse_args())
    return args


def write_result_on_image(src_img, res, box):
    y, x, h, w = box
    color = (255, 255, 0)
    cv2.rectangle(src_img, (round(x), round(y)), (round(w), round(h)), color, 2)
    cv2.putText(src_img, res, (round(x) - 10, round(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def limit_to_one(class_ids, boxes, confidences):
    class_id = None
    box = None
    confidences = None
    return

# 圖片、預測類別、box大小
def get_code_and_draw(frame, crop_img, re_box,predicted_class_2, debug=False):
    """
    Detect the code in the frame and annotate it.
    """
    codes = []
    #--------------------------------------#
    #將照片所框的範圍進行擷取並將其等比例擴大
    #--------------------------------------#
    crop_img = resize_to_suitable(crop_img)

    print("resized image shape", crop_img.shape)
    if debug:
        display_image_cv2(crop_img, "cropped code image")
    #對照片進行前處理   
    clean_img = process_image_for_ocr(crop_img, debug)
    # 將黑底白字照片轉乘白底黑字
    clean_img = cv2.bitwise_not(clean_img)
    # 使用影像平滑模糊化消除雜訊，主要是透過使用低通濾波器進行影像卷積來實現。
#         clean_img = cv2.blur(clean_img, (2, 2))
    # 丟入ocr內進行影像辨識
    res = ocr.find_code_in_image(clean_img,predicted_class_2)
    print("Detected code:", res)
    codes.append(res)
    # 將文字寫在照片上    
    write_result_on_image(frame, res, re_box)
    if debug:
        display_image_cv2(clean_img, "cleaned code image")
    return frame, codes

def get_code(frame, class_ids, classes, boxes, debug=False):
    """
    Detect the code in the frame and annotate it.
    """
    n_obj = len(class_ids)
    codes = []
    # Crop the code part out and process it
    for i in range(n_obj):
        if classes[class_ids[i]] == "sidecode":
            print("single_frame_process/get_code_and_draw(): sidecode not working yet!")
            continue
        x, y, w, h = boxes[i]
        extra_pix = 2  # Get a bit bigger crop to make sure we cover everything
        crop_img = frame[round(y)-extra_pix:round(y+h)+extra_pix, round(x)-extra_pix:round(x+w)+extra_pix]
        print("single_frame_process/get_code_and_draw(): cropped image shape", crop_img.shape)
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            continue
        crop_img = resize_to_suitable(crop_img)
        print("single_frame_process/get_code_and_draw(): resized image shape", crop_img.shape)
        if debug:
            display_image_cv2(crop_img, "cropped code image")
        clean_img = process_image_for_ocr(crop_img, debug)
        clean_img = cv2.bitwise_not(clean_img)
        clean_img = cv2.blur(clean_img, (2, 2))
        
        res = ocr.find_code_in_image(clean_img)
        print("Detected code:", res)
        codes.append(res)
        if debug:
            display_image_cv2(clean_img, "cleaned code image")
    return codes

def get_code_and_draw_true(frame, class_ids, classes, boxes, code, debug=False):
    """
    Detect the code in the frame and annotate it.
    """
    n_obj = len(class_ids)
    codes = []
    # Crop the code part out and process it
    for i in range(n_obj):
        if classes[class_ids[i]] == "sidecode":
            print("single_frame_process/get_code_and_draw(): sidecode not working yet!")
            continue
        x, y, w, h = boxes[i]
        extra_pix = 2  # Get a bit bigger crop to make sure we cover everything
        crop_img = frame[round(y)-extra_pix:round(y+h)+extra_pix, round(x)-extra_pix:round(x+w)+extra_pix]
        print("single_frame_process/get_code_and_draw(): cropped image shape", crop_img.shape)
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            continue
        crop_img = resize_to_suitable(crop_img)
        print("single_frame_process/get_code_and_draw(): resized image shape", crop_img.shape)
        if debug:
            display_image_cv2(crop_img, "cropped code image")
        clean_img = process_image_for_ocr(crop_img, debug)
        clean_img = cv2.bitwise_not(clean_img)
        clean_img = cv2.blur(clean_img, (2, 2))
        
        res = code
        print("Detected code:", code)
#         codes.append(res)
#         if debug:
#             display_image_cv2(clean_img, "cleaned code image")
        write_result_on_image(frame, res, boxes[i])
    return frame, codes

def main(args):
    """
    If run this file, it will perform code region detection and OCR it on an image instead of a video.
    """
    src_img = cv2.imread(args['image'])
    # Build model and detect code region
    net, classes, output_layers = build_model(args['classes'], args['weights'], args['config'])
    # For some reason, enable CUDA make single image detect slower?
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    class_ids, boxes, confidences = detect(net, src_img, output_layers)
    src_img, _ = get_code_and_draw(src_img, class_ids, classes, boxes, debug=True)
    display_image_cv2(src_img, "final")
    cv2.imwrite("output.jpg", src_img)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    argument = argument_parser()
    main(argument)