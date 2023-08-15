import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def error_check(formatted_code,predicted_class_2):
    """ Assume code is using BIC container code format. """
    fixed_code = list(formatted_code)
    n = len(fixed_code)
#     print("**n**", n)
    if n < 15:
        print("Code is not complete!")
        n = ""
        return n
    else:
        # First 4 characters are always letters
        for i in range(4):
            if fixed_code[i] == '1':
                fixed_code[i] = 'I'
            if fixed_code[i] == '4':
                fixed_code[i] = 'A'
            if fixed_code[i] == '6':
                fixed_code[i] = 'G'
            if fixed_code[i] == '8':
                fixed_code[i] = 'B'
        # The next 6 characters are always digits
        for i in range(4, 10):
            if fixed_code[i] == 'I':
                fixed_code[i] = '1'
            if fixed_code[i] == 'A':
                fixed_code[i] = '4'
            if fixed_code[i] == 'G':
                fixed_code[i] = '6'
            if fixed_code[i] == 'B':
                fixed_code[i] = '8'
    fixed_code = check_code(fixed_code,predicted_class_2)              
    fixed_code = "".join(fixed_code)
    
    return fixed_code


def reformat_code(original_code):
    """ Reformat the text into better format. Format assumed to be BIC container code. """
    formatted_code = None
    print('original_code:',original_code)
    n = len(original_code)
    if n <= 100:
        # Likely to be backcode
        formatted_code = original_code[0:n - 1]  # Remove weird last character
        formatted_code = formatted_code.replace(" ", "")  # Remove whitespace characters
        formatted_code = formatted_code.replace("\n", "")  # Remove whitespace character
    return formatted_code


def build_tesseract_options(is_sidecode=False):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "--psm 3"  # Fully automatic page segmentation, but no OSD. (Default)
    options += " --oem 3"
    options += " -c tessedit_char_whitelist={}".format(alphanumeric)
    options += " load_freq_dawg=false load_system_dawg=false"  # Tesseract won't use dictionary
    # set the PSM mode
    # options += " --psm {}".format(7)
    # return the built options string
    return options


def find_code_in_image(img,predicted_class_2):
    im_h, im_w = img.shape[0:2]
    options = build_tesseract_options()
    #辨識圖片
    result = pytesseract.image_to_string(img, config=options)
    #輸出文字中有空格之類的符號將其移除
    result = reformat_code(result)
    print(result)
    # 根據號碼進行後處理
    result = error_check(result,predicted_class_2)
    print("**result**", result)
    return result

def numSplit(num):
    '''
    浮點數字整數、小數分離【將數字轉化爲字符串處理】
    '''
    zs,xs=str(num).split('.')
    return xs[0]

def check_code(num,predicted_class_2):
    dic ={
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18, "I": 19, "J": 20, "K": 21
    , "L": 23, "M": 24, "N": 25, "O": 26, "P": 27, "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34
    , "W": 35, "X": 36, "Y": 37, "Z": 38, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6
    , "7": 7, "8": 8, "9": 9
    }
    lst = []
    for i in num:
        lst.append(dic[i])
    formula1 = lst[0]*1 + lst[1]*2 + lst[2]*4 + lst[3] * 8 + lst[4] * 16 + lst[5] * 32 + lst[6] * 64 + lst[7] * 128 + lst[8] * 256 + lst[9]*512
    formula2 = formula1%11
    
    if(formula2 == 10 or formula2 == 0):
        formula2 = 0
    else:
        formula2 = formula2
     
    if (int(predicted_class_2[0]) == formula2):
        container_num =""
        for i in range(0,10): 
            if(i==9):
                container_num += num[i]+predicted_class_2[0]
            else:
                container_num += num[i]
        return container_num
    else:
        return ""


def main():
    ...


if __name__ == '__main__':
    main()

# TODO: Improve reformat for side code
# TODO: Use ISO_6346 check digits for better accuracy
# https://en.wikipedia.org/wiki/ISO_6346#Check_digit
# https://github.com/arthurdejong/python-stdnum/blob/master/stdnum/iso6346.py
