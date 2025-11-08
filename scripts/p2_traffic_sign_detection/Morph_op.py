
import cv2
import numpy as np

def BwareaOpen(img, MinArea):
    # Bước 1: Nhị phân hóa
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    
    # Bước 2: Tìm contour
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return thresh

    # Bước 3: Xóa contour nhỏ hơn MinArea
    cnts_TooSmall = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < MinArea:
            cnts_TooSmall.append(cnt)

    cv2.drawContours(thresh, cnts_TooSmall, -1, 0, -1)
    return thresh
