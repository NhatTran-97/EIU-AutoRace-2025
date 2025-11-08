import cv2
import os
from Signs.Localization.UsingHough import detect_Circles

from Signs.Localization.UsingApproxpolyDP import LocalizeSigns

def main():
    cap = cv2.VideoCapture(os.path.abspath("data/signs_forward.mp4"))

    if not cap.isOpened():
        print("❌ Không mở được video!")
        return

    waitTime = 30  # 30 ms tương đương ~33 FPS

    while True:
        ret, img = cap.read()
        if not ret:
            print("Video đã chạy hết.")
            break

        img = cv2.resize(img, (320, 240))
        # img_draw = img.copy()

        CirclesUsingApproxPoly = LocalizeSigns(img)

        # detect_Circles(img, img_draw)

        # Hiển thị khung hình
        # cv2.imshow("Detected Circles", img_draw)

        # Nhấn ESC để thoát
        k = cv2.waitKey(waitTime) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
