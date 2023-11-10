import numpy as np
import cv2 as cv
import easyocr

reader = easyocr.Reader(['en'])

# 將圖片灰階化再做反白
def img_process(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11,5)    
    min_area_threshold = 100
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] < min_area_threshold:
            labels[labels == i] = 0
    cleaned_thresh = (labels > 0).astype(np.uint8) * 255
    
    return cleaned_thresh

# 將圖片裁切多餘的部份，只留下車牌
def crop_lp(original_img):
    # 放大比較好辨識
    img = cv.resize(original_img, None, fx = 3, fy = 3, interpolation = cv.INTER_CUBIC)
    h, w, c = img.shape

    if w > 276:
        x1 = int(0.125 * w)
        x2 = int(0.88 * w)
    else:
        x1 = int(0.125 * w)
        x2 = int(0.9 * w)
    y1 = int(0.3 * h)
    y2 = int(0.78 * h)
    crop_img = img[y1:y2, x1:x2]
    return crop_img

# 執行單一車牌號碼辨識
def character_recognize(img):
    result = reader.readtext(img)
    filtered_results = []
    for detection in result:
        text, prob = detection[1], detection[2]
        if len(text) > 1:
            text = text[0]
        filtered_text = ''.join([char for char in text if char.isalnum()])
        filtered_results.append((filtered_text, prob))

    final_results = []
    for text, prob in filtered_results:
        if 'g' in text:  # 检查文本中是否包含字母"g"
            text = text.replace('g', '9')  # 将"g"替换为"9"                
        text = text.upper() 
        final_results.append((text, prob))

    # 打印過濾後的結果
    # for text, prob in final_results:
    #     print(f'Text: {text}, Confidence: {prob:.8f}')

    return final_results

def lp_recognize(img):
    result = reader.readtext(img)
    filtered_results = []
    for detection in result:
        text, prob = detection[1], detection[2]
        filtered_text = ''.join([char for char in text if char.isalnum()])
        filtered_results.append((filtered_text, prob))

    final_results = []
    for text, prob in filtered_results:
        if len(text) == 7:
            if text[0] in ['I', 'J']:
                text = text[1:]
            elif text[-1] in ['I', '1']:
                text = text[:-1]
            else:
                text = text[1:]
        if len(text) >= 8:
            text = text[1:-1]
        
        if 'g' in text:  # 检查文本中是否包含字母"g"
            text = text.replace('g', '9')  # 将"g"替换为"9"
        
        situation = 0
        if not any(char.isalpha() for char in text)  and len(text) == 6:  # 检查字符串中是否没有任何字母且長度為6
            for index in [1, 5]:  # 检查特定位置的字符
                if text[index] == '8':  # 如果字符为'0'
                    text = text[:index] + 'B' + text[index + 1:]  # 将'0'替换为'D
                    situation = 1

            if situation == 0:
                for index in [0, 1, 4, 5]:  # 检查特定位置的字符
                    if text[index] == '0':  # 如果字符为'0'
                        text = text[:index] + 'D' + text[index + 1:]  # 将'0'替换为'D
                    
        text = text.upper() 
        final_results.append((text, prob))

    # 打印過濾後的結果
    for text, prob in final_results:
        print(f'Text: {text}, Confidence: {prob:.8f}')
    # cv.namedWindow('window', cv.WINDOW_NORMAL)
    # cv.resizeWindow('window', 800, 600)
    # cv.imshow('window', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return text

def check_all_same(my_list):
    return all(x == my_list[0] for x in my_list)