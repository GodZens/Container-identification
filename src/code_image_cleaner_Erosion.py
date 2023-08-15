import cv2
import numpy as np
from src.utils import display_image_cv2

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
    brightness = -(img.mean()*2/5)
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

#照片二值化
def otsu_threshold(src_img,debug):
    """ NOT expected to return white text on black background"""
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    #侵蝕膨脹    
    kernel = np.ones((2,2), np.uint8)
    erosion = cv2.erode(gray_img, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    # 自適應直方圖均衡化
#     clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))

#     clahe = clahe.apply(gray_img)
#     blur_img, sharp_image = sharpen(gray_img)

    # blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh_img1 = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     thresh_img2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     thresh_img = cv2.bitwise_and(thresh_img1, thresh_img2)
    if debug:
        display_image_cv2(erosion, "erosion")
        display_image_cv2(dilation, "dilation")
        
    return thresh_img1


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
