import cv2
import numpy as np
from Morph_op import BwareaOpen
from estimation_al import Estimate_MidLane
from ExtendLanesAndRefineMidLaneEdge import ExtendShortLane
import config

# === Global variables ===
HLS = None
src = None

# ========================== BIRDVIEW TUNER ====================================
src_points = np.float32([[0,0],[0,0],[0,0],[0,0]])
dragging_idx = -1

def mouse_birdview(event, x, y, flags, param):
    """Cho phép click và kéo 4 điểm tuning trực tiếp"""
    global dragging_idx, src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(src_points):
            if np.linalg.norm(p - np.array([x, y])) < 10:
                dragging_idx = i
    elif event == cv2.EVENT_MOUSEMOVE and dragging_idx != -1:
        src_points[dragging_idx] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = -1

def birdview_transform_tune(img):
    """Hiển thị BirdView + cho phép kéo trực tiếp 4 điểm"""
    global src_points
    h, w = img.shape[:2]

    if np.all(src_points == 0):
        # init 4 điểm mặc định
        src_points = np.float32([
            [w*0.10, h*0.95],   # bottom-left
            [w*0.90, h*0.95],   # bottom-right
            [w*0.65, h*0.55],   # top-right
            [w*0.35, h*0.55]    # top-left
        ])

    cv2.namedWindow("BirdView Region", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("BirdView Region", mouse_birdview)

    # Vẽ polygon
    overlay = img.copy()
    for p in src_points.astype(int):
        cv2.circle(overlay, tuple(p), 6, (0,0,255), -1)
    cv2.polylines(overlay, [src_points.astype(int)], True, (0,255,0), 2)
    cv2.imshow("BirdView Region", overlay)

    # Warp ảnh sang birdview
    dst = np.float32([
        [200, h],
        [w-200, h],
        [0, 0],
        [w, 0]
    ])
    M = cv2.getPerspectiveTransform(src_points, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    cv2.imshow("BirdView", warped)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('s'):
        print("💾 Saved src_points =", src_points.tolist())
    elif key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    return warped

# ========================== MASK & ROI ====================================

def clr_segment(HLS, lower_range, upper_range):
    lower = np.array(lower_range, dtype=np.uint8)
    upper = np.array(upper_range, dtype=np.uint8)
    mask = cv2.inRange(HLS, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

def LaneROI(frame, mask, minArea):
    frame_Lane = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("frame_Lane", frame_Lane)
    Lane_gray = cv2.cvtColor(frame_Lane, cv2.COLOR_BGR2GRAY)
    Lane_gray_opened = BwareaOpen(Lane_gray, minArea)
    Lane_gray = cv2.bitwise_and(Lane_gray, Lane_gray_opened)
    Lane_gray_Smoothed = cv2.GaussianBlur(Lane_gray, (11,11), 1)
    Lane_edge = cv2.Canny(Lane_gray_Smoothed, 50, 150, None, 3)
    return Lane_edge, Lane_gray_opened

# ========================== LANE DETECTION ====================================

Hue_Low, Lit_Low, Sat_Low = 0, 160, 0
Hue_High, Lit_High, Sat_High = 180, 255, 90

def find_lane_lines(img):
    global HLS, src
    src = img.copy()
    HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    cv2.imshow("HLS", HLS)

    mask_W = clr_segment(HLS, (Hue_Low, Lit_Low, Sat_Low), (255, 255, 255))
    Mid_edge_ROI, Mid_ROI_mask = LaneROI(img, mask_W, 500)
    cv2.imshow("Mid_edge_ROI", Mid_edge_ROI)
    cv2.imshow("Mid_ROI_mask", Mid_ROI_mask)
    return Mid_edge_ROI

# ========================== CONTROL ====================================

def calculate_control_signal(img, draw=None):
    """Main control pipeline"""
    img_lines = find_lane_lines(img)
    img_birdview = birdview_transform_tune(img_lines)
    cv2.imshow("Lane Lines", img_lines)
    cv2.imshow("BirdView", img_birdview)
    return 0,0
