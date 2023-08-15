import cv2
import numpy as np
from src.utils import display_image_cv2
from PIL import Image
#對圖片進行旋轉
def rotate_image(thresh_img, debug=False):
    """ Rotate an image. Required input to be a binary image."""
    if debug:
        display_image_cv2(thresh_img, "prerotate")
    im_h, im_w = thresh_img.shape[0:2]
    if im_h > im_w:
        # Not yet implemented for vertical side code
        print("code_image_cleaner/rotate_image(): Not yet implemented for vertical side code")
        return thresh_img

    tmp = np.where(thresh_img > 0)
    row, col = tmp
    # note: column_stack is just vstack().T (aka transposed vstack)
    coords = np.column_stack((col, row))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if debug:
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        debug_box_img = cv2.drawContours(thresh_img.copy(), [box_points], 0, (255, 255, 255), 2)
        display_image_cv2(debug_box_img, "debug box rotate", False)
    # the v4.5.1 `cv2.minAreaRect` function returns values in the
    # range (0, 90]); as the rectangle rotates clockwise the
    # returned angle approach 90.
    if angle > 45:
        # if angle > 45 it will rotate left 90 degree into vertical standing form, so rotate another 270 degree
        # will bring it back to good. Otherwise, it will rotate nice.
        angle = 270 + angle

    # rotate the image
    (h, w) = thresh_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    print("code_image_cleaner/rotate_image(): rotated", angle)
    return rotated
        
def is_contour_bad(c, src_img):
    im_h, im_w = src_img.shape[0:2]
    box = cv2.boundingRect(c)
    x, y, w, h = box[0], box[1], box[2], box[3]
    # If image is a back code (width larger than height)
    if im_w > im_h:
        if h >= 0.6*im_h:  # likely to be a bar
            print("found a bar contour")
            return True
        if x < 0.05*im_w and y < 0.05*im_h:  # 左上區域不會有文字
            print("found a unrelated contour")
            return True
        if x < 0.05*im_w and y > 0.95*im_h:  # 左下區域不會有文字
            print("found a unrelated contour")
            return True
        if x > 0.95*im_w and y < 0.05*im_h:  # 右上區域不會有文字
            print("found a unrelated contour")
            return True
        if x > 0.95*im_w and y > 0.95*im_h:  # 右下區域不會有文字
            print("found a unrelated contour")
            return True
        if w*h < 0.002*im_h*im_w:  # Noise w/ area < 0.2% of image's area
            print("found a tiny noise contour")
            return True
        if x <= 1 or x >= (im_w-1) or y <= 1 or y >= (im_h-1):
            if w*h < 0.005*im_h*im_w:
                print(x, y, w, h, im_w, im_h)
                print("code_image_cleaner/is_contour_bad(): found a sus edge-touched contour")
                return True
    return False

#拐點位置、二值化圖片、原始照片
def remove_noise(cnts, thresh, src_img, debug=False):
    print("===Start removing noise===")
    # 將規一化照片轉成正常照片數值255
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    # loop over the contours
    for c in cnts:
        # Draw contour for visualization
        if debug:
            #將拐點訊息丟入將返回[ x, y, w, h]及一個框住物體的矩型資訊
            box = cv2.boundingRect(c)
            x, y, w, h = box[0], box[1], box[2], box[3]
            # 根據這個矩型資訊劃出
            cv2.rectangle(src_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # 查找不符合文字的矩陣，若出現則將其填充成背景顏色
        if is_contour_bad(c, src_img):
            cv2.drawContours(mask, [c], -1, 0, -1)
    # remove the contours from the image and show the resulting images
    #逐位元and邏輯運算移除輪廓
    result = cv2.bitwise_and(thresh, thresh, mask=mask)
    print("=====Finish=====")
    return result
#降低反光
def unlight(img):
    contrast = 0
    brightness = -(img.mean()*3/5)
    output = img * (contrast/127 + 1) - contrast + brightness # 轉換公式
    # 轉換公式參考 https://stackoverflow.com/questions/50474302/how-do-i-adjust-brightness-contrast-and-vibrance-with-opencv-python

    # 調整後的數值大多為浮點數，且可能會小於 0 或大於 255
    # 為了保持像素色彩區間為 0～255 的整數，所以再使用 np.clip() 和 np.uint8() 進行轉換
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    return output
#銳化
def sharpen(gray_img):    
    blur_img = cv2.blur(gray_img, (7, 7))
    usm = cv2.addWeighted(gray_img, 1.5, blur_img, -0.5, 0)
    
    return blur_img,usm

def gray_bgr(Re_image,src_img):
    B = src_img[:,:,0]
    G = src_img[:,:,1]
    R = src_img[:,:,2]
    g = Re_image[:]
    p=0.2989;q=0.5870;t=0.1140
    new_B = (g-p*R-q*G)/t
    new_B = np.uint8(new_B)
    new_src = np.zeros((src_img.shape)).astype("uint8")
    new_src[:,:,0] = R
    new_src[:,:,1] = G
    new_src[:,:,2] = new_B
    return new_src

#照片二值化
def otsu_threshold(src_img,debug):
    """ NOT expected to return white text on black background"""
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img_size = gray_img.shape
    h = img_size[0]
    w = img_size[1]
    row_index = 0
    #查看圖片分割點位置
    for i in range(0,1):
        for j in range(w):
            #右>左
            if(gray_img[i,w-1]>gray_img[i,0]):
                if(gray_img[i,j]>int(gray_img[i,:].mean()*1.2)):
                    row_index = j
                    break;
            else:
                if(gray_img[i,w-j-1]>int(gray_img[i,:].mean()*1.2)):
                    row_index = w-j-1
                    break;
    #查看該分割點是否會切到數字
    for j in range(row_index,w):
        col = gray_img[0:h,j]
        gap = col.max()-col.min()
        if(abs(gap)<int(gray_img.mean())):
            row_index = j
            break;        
    #將其切成兩部分
    right = gray_img[0:h,row_index:w]
    left = gray_img[0:h,0:row_index]
    left_light = left.mean()
    right_light = right.mean()
    print('left_light：',left_light)
    print('right_light：',right_light)
    #查看要用哪一種二質化公式
    equal_img = cv2.equalizeHist(gray_img)
    originalhold = np.mean(equal_img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray_img) 
    _, threshold_image = cv2.threshold(clahe_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = np.mean(threshold_image)
    print('threshold：',threshold)
    print('originalhold：',originalhold)
    if(threshold>60 and originalhold > 127):
        if(abs(left_light-right_light) > 10 ):
            left_dilated_img = cv2.dilate(left, np.ones((7,7), np.uint8)) 
            left_bg_img = cv2.medianBlur(left_dilated_img, 29)
            left_diff_img = 255 - cv2.absdiff(left, left_bg_img)
            left_blur_img,left_sharpen_img = sharpen(left_diff_img)
            _, left_thresh_img = cv2.threshold(left_sharpen_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            make_sure_it_bbwt(left_thresh_img)

            right_dilated_img = cv2.dilate(right, np.ones((7,7), np.uint8)) 
            right_bg_img = cv2.medianBlur(right_dilated_img, 29)
            right_diff_img = 255 - cv2.absdiff(right, right_bg_img)
            right_blur_img,right_sharpen_img = sharpen(right_diff_img)
            _, right_thresh_img = cv2.threshold(right_sharpen_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            make_sure_it_bbwt(right_thresh_img)
            
            Re_image = cv2.hconcat([left_sharpen_img,right_sharpen_img])
            thresh_img_1 = cv2.hconcat([left_thresh_img,right_thresh_img])
            
            if debug:
                display_image_cv2(left_diff_img, "left_diff_img")
                display_image_cv2(right_diff_img, "right_diff_img")
                display_image_cv2(left_thresh_img, "left_thresh_img")
                display_image_cv2(right_thresh_img, "right_thresh_img")
                
        else:
            dilated_img = cv2.dilate(gray_img, np.ones((7,7), np.uint8)) 
            bg_img = cv2.medianBlur(dilated_img, 29)
            diff_img = 255 - cv2.absdiff(gray_img, bg_img)
            blur_img,sharpen_img = sharpen(diff_img)
            Re_image = sharpen_img
            _, thresh_img_1 = cv2.threshold(sharpen_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            make_sure_it_bbwt(thresh_img_1)
            
    else:
        #若高亮度圖片遠大於低亮度圖，則降低光亮
        if(abs(left_light-right_light) > 10 ):
            if(left_light>right_light):
                #降低反光
                left_unlight = unlight(left)
                #自是應對比
                left_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                left_clahe_img = left_clahe.apply(left_unlight) 
                left_blur_img,left_sharpen_img = sharpen(left_clahe_img)
                kernel = np.ones((2,2), np.uint8)
                left_erosion = cv2.erode(left_sharpen_img, kernel, iterations = 1)
                left_dilation = cv2.dilate(left_erosion, kernel, iterations = 1)
                
                #右邊
                right_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                right_clahe_img = right_clahe.apply(right) 
                right_blur_img,right_sharpen_img = sharpen(right_clahe_img)
                kernel = np.ones((2,2), np.uint8)
                right_erosion = cv2.erode(right_sharpen_img, kernel, iterations = 1)
                right_dilation = cv2.dilate(right_erosion, kernel, iterations = 1)               
                
                Re_image = cv2.hconcat([left_dilation,right_dilation])
                _, thresh_img_1 = cv2.threshold(Re_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                if debug:
                    display_image_cv2(left_unlight, "left_unlight")
                    display_image_cv2(left_clahe_img, "left_clahe_img")
                    display_image_cv2(left_blur_img, "left_blur_img")
                    display_image_cv2(left_erosion, "left_erosion")
                    display_image_cv2(left_dilation, "left_dilation")
                    display_image_cv2(right_clahe_img, "right_clahe_img")
                    display_image_cv2(right_blur_img, "right_blur_img")
                    display_image_cv2(right_erosion, "right_erosion")
                    display_image_cv2(right_dilation, "right_dilation")
            else:
                #降低反光
                right_unlight = unlight(right)
                right_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                right_clahe_img = right_clahe.apply(right_unlight)    
                right_blur_img,right_sharpen_img = sharpen(right_clahe_img)
                kernel = np.ones((2,2), np.uint8)
                right_erosion = cv2.erode(right_sharpen_img, kernel, iterations = 1)
                right_dilation = cv2.dilate(right_erosion, kernel, iterations = 1)
                
                left_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                left_clahe_img = left_clahe.apply(left) 
                left_blur_img,left_sharpen_img = sharpen(left_clahe_img)
                kernel = np.ones((2,2), np.uint8)
                left_erosion = cv2.erode(left_sharpen_img, kernel, iterations = 1)
                left_dilation = cv2.dilate(left_erosion, kernel, iterations = 1)     
                
                Re_image = cv2.hconcat([left_dilation,right_dilation])
                _, thresh_img_1 = cv2.threshold(Re_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                if debug:
                    display_image_cv2(right_unlight, "right_unlight")
                    display_image_cv2(left_clahe_img, "left_clahe_img")
                    display_image_cv2(left_blur_img, "left_blur_img")
                    display_image_cv2(left_erosion, "left_erosion")
                    display_image_cv2(left_dilation, "left_dilation")
                    display_image_cv2(right_clahe_img, "right_clahe_img")
                    display_image_cv2(right_blur_img, "right_blur_img")
                    display_image_cv2(right_erosion, "right_erosion")
                    display_image_cv2(right_dilation, "right_dilation")

        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray_img) 
            blur_img,sharpen_img = sharpen(clahe_img)
            kernel = np.ones((2,2), np.uint8)
            erosion = cv2.erode(sharpen_img, kernel, iterations = 1)
            dilation = cv2.dilate(erosion, kernel, iterations = 1)
            Re_image = dilation
        
            _, thresh_img_1 = cv2.threshold(Re_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if debug:
        display_image_cv2(Re_image, "Re_image")
        display_image_cv2(thresh_img_1, "thresh_img_1")
       
    return thresh_img_1


def make_bbwt(thresh_img, depth=2):
    """ Make sure the thresh img has white text on black background """
    im_h, im_w = thresh_img.shape[0:2]
    # Calculate the pixel value of image border
    total_pixel_value = np.sum(thresh_img)
    center_img = thresh_img[depth:im_h-depth, depth:im_w-depth]
    center_pixel_value = np.sum(center_img)
    #獲得邊框平均像素
    border_bw_value = (total_pixel_value - center_pixel_value) / (im_h*im_w - center_img.size)
    print("code_image_cleaner/is_it_bbwt():BBWT value:", border_bw_value)
    # If True mean it is not bbwt, and thresh must be invert
    # 若邊框顏色趨近白底，則將其改成黑底
    if border_bw_value > 127:
        return False
    else:
        return True


def make_sure_it_bbwt(thresh_img, depth=2):
    """ Make sure the thresh img has white text on black background """
    im_h, im_w = thresh_img.shape[0:2]
    # Calculate the pixel value of image border
    total_pixel_value = np.sum(thresh_img)
    center_img = thresh_img[depth:im_h-depth, depth:im_w-depth]
    center_pixel_value = np.sum(center_img)
    #獲得邊框平均像素
    border_bw_value = (total_pixel_value - center_pixel_value) / (im_h*im_w - center_img.size)
    print("code_image_cleaner/is_it_bbwt():BBWT value:", border_bw_value)
    # If True mean it is not bbwt, and thresh must be invert
    # 若邊框顏色趨近白底，則將其改成黑底
    if border_bw_value > 127:
        cv2.bitwise_not(thresh_img, thresh_img)

#對照片進行前處理
def process_image_for_ocr(src_img, debug=False):
    """
    Clean up other cluttering on the back code and return a binary image. Run this from other file.
    """
    # Binarization
    #照片二值化
    thresh = otsu_threshold(src_img,debug)
    make_sure_it_bbwt(thresh)
    # 只检测圖片最外围轮廓，並仅保存轮廓的拐点信息
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Remove noise(去雜訊)
    clean = remove_noise(cnts, thresh, src_img, debug)

#     # Rotate(圖片進行旋轉)
#     rotated = rotate_image(clean)
    if debug:
        display_image_cv2(src_img, "original w/ box")
        display_image_cv2(thresh, "thresh")
        display_image_cv2(clean, "removed noise")
#         display_image_cv2(rotated, "rotated")
    return clean


def main():
    """ Test threshold and cleanup ability """
    src_img = cv2.imread("../images/code5_fix.png")
    clean_img = process_image_for_ocr(src_img, True)


if __name__ == "__main__":
    main()
