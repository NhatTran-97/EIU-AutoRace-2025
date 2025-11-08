import cv2
import numpy as np
from Morph_op import BwareaOpen

import config
import numpy as np
import cv2
import numpy as np
from skimage.morphology import skeletonize

# === Global variables ===

HLS=None
src=None

# Global biến để lưu 4 điểm và trạng thái kéo
src_points = np.float32([[0,0],[0,0],[0,0],[0,0]])
dragging_idx = -1
tuning_enabled = True   # để tắt/bật tuning khi chạy thật

bw_params = {
    "top_y": 55,     # phần trăm chiều cao (%)
    "bot_y": 97,
    "left_x": 10,
    "right_x": 90
}

import cv2
import numpy as np

# Hue_Low, Lit_Low, Sat_Low = 82, 95, 60
Hue_Low, Lit_Low, Sat_Low = 0, 160, 0     # màu trắng sáng (Lightness cao, Saturation thấp)
Hue_High, Lit_High, Sat_High = 180, 255, 90  # tránh bị lấy cỏ hoặc vàng


# === Callback khi thay đổi trackbar ===

def OnHueLowChange(v):  global Hue_Low; MaskExtract()
def OnLitLowChange(v):  global Lit_Low; MaskExtract()
def OnSatLowChange(v):  global Sat_Low; MaskExtract()

def nothing(x):
    pass


def MaskExtract():
    global HLS, src
    if HLS is None or src is None: return
    mask = clr_segment(HLS, (Hue_Low, Lit_Low, Sat_Low), (Hue_High, Lit_High, Sat_High))
    mask_rgb = cv2.merge([mask, mask, mask])
    dst = cv2.bitwise_and(src, mask_rgb)
    

def clr_segment(HLS, lower_range, upper_range):
    lower = np.array(lower_range, dtype=np.uint8)
    upper = np.array(upper_range, dtype=np.uint8)
    mask = cv2.inRange(HLS, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel) # lam muot va noi lien vung lane

    return mask # MASK chua vung lane trang da trich xuat khoi nen

def LaneROI(frame,mask,minArea):
    
    # 4a. Keeping only ROI of frame
    frame_Lane = cv2.bitwise_and(frame,frame,mask=mask)#Extracting only RGB from a specific region/ pixel =255 duoc giu lai
    cv2.imshow("frame_Lane: ", frame_Lane)
    # 4b. Converting frame to grayscale
    Lane_gray = cv2.cvtColor(frame_Lane,cv2.COLOR_BGR2GRAY) # Converting to grayscale
    # 4c. Keep Only larger objects
    Lane_gray_opened = BwareaOpen(Lane_gray,minArea) # Getting mask of only objects larger then minArea
    
    Lane_gray = cv2.bitwise_and(Lane_gray,Lane_gray_opened)# Getting the gray of that mask



    Lane_gray_Smoothed = cv2.GaussianBlur(Lane_gray,(11,11),1) # Smoothing out the edges for edge extraction later
    # 4d. Keeping only Edges of Segmented ROI    
    Lane_edge = cv2.Canny(Lane_gray_Smoothed,50,150, None, 3) # Extracting the Edge of Canny

  
    return Lane_edge,Lane_gray_opened


def extract_lane_points(edge_img):
    """
    Từ ảnh edge (đen trắng) -> danh sách tọa độ pixel trắng.
    """
    points = np.column_stack(np.where(edge_img > 0))  # [y, x]
    # Đảo thứ tự thành [(x, y)]
    points = [(int(x), int(y)) for y, x in points]
    return points

def extract_lane_contours(edge_img):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # sort theo X (trái/phải)

    lanes = []
    for i, c in enumerate(contours):
        pts = c.squeeze()  # [[x,y], ...]
        lanes.append(pts)
        color = (255, 0, 0) if i == 0 else (0, 0, 255)
        cv2.drawContours(edge_img, [c], -1, color, 2)
    return lanes




def find_lane_lines(img):
    """
    Detecting road markings
    This function will take a color image, in BGR color system,
    Returns a filtered image of road markings
    """

    global HLS, src
    src = img.copy()
    HLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    cv2.imshow("HLS:", HLS)
    mask_W   = clr_segment(HLS,(Hue_Low, Lit_Low, Sat_Low ),(255, 255, 255))

    
    Mid_edge_ROI,Mid_ROI_mask = LaneROI(img,mask_W,500)#20 msec


    cv2.imshow("Mid_edge_ROI:", Mid_edge_ROI)
    # Return image
    return Mid_ROI_mask

def birdview_transform(img):
    """Apply bird-view transform to the image
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    cv2.imshow("warped_img: ", warped_img)
    return warped_img





def find_left_right_points(image, draw=None):
    """Find left and right points of lane
    """

    im_height, im_width = image.shape[:2]

    # Consider the position 70% from the top of the image
    interested_line_y = int(im_height * 0.9)
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (im_width, interested_line_y), (0, 0, 255), 2)
    interested_line = image[interested_line_y, :]

    # Detect left/right points
    left_point = -1
    right_point = -1
    lane_width = 100
    center = im_width // 2

    # Traverse the two sides, find the first non-zero value pixels, and
    # consider them as the position of the left and right lines
    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    # Predict right point when only see the left point
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width

    # Predict left point when only see the right point
    if right_point != -1 and left_point == -1:
        left_point = right_point - lane_width

    # Draw two points on the image
    if draw is not None:
        if left_point != -1:
            draw = cv2.circle(
                draw, (left_point, interested_line_y), 7, (255, 255, 0), -1)
        if right_point != -1:
            draw = cv2.circle(
                draw, (right_point, interested_line_y), 7, (0, 255, 0), -1)

    return left_point, right_point





def calculate_control_signal(img, draw=None):
    """Calculate speed and steering angle
    """

    # Find left/right points
    img_lines = find_lane_lines(img)

    warped = birdview_transform(img_lines)

    cv2.imshow("lane_single: ", warped)
    mid_points = extract_midline_from_mask(warped)
    visualize_midline(warped, mid_points)



#     bev_vis = cv2.cvtColor(lane_single, cv2.COLOR_GRAY2BGR)
#   #  midpoints = extract_midline_points(lane_single)
#     midpoints = extract_lane_midline(warped)
#     for (x, y) in midpoints:
#         cv2.circle(bev_vis, (x, y), 2, (0, 255, 0), -1)

#     cv2.imshow("Merged Lane", lane_single)
#     cv2.imshow("BEV Visualization", bev_vis)
   # lane_single[:, :] = birdview_transform(draw)

    # lanes = extract_lane_contours(warped.copy())
    # if len(lanes) < 2:
    #     print("⚠️ Không tìm thấy đủ 2 lane, bỏ qua frame này")
    #     return 0.0, 0.0
    
    # print("lanes: ", lanes)
    # mid_points = extract_midlane_points(warped)
    # print("✅ Số điểm midlane:", len(mid_points))
    left_point, right_point = find_left_right_points(warped, draw=draw)


    # sliding_out = sliding_window_dual_lane(warped,  nwindows=9, margin=50, minpix=50)
    # cv2.imshow("Sliding Window Lane", sliding_out)


    throttle = 0.5
    steering_angle = 0
    im_center = img.shape[1] // 2

    if left_point != -1 and right_point != -1:

        # Calculate the deviation
        center_point = (right_point + left_point) // 2
        center_diff =  im_center - center_point

        # Calculate steering angle
        # You can apply some advanced control algorithm here
        # For examples, PID
        steering_angle = - float(center_diff * 0.01)


    #return throttle, steering_angle
    return 0, 0


def extract_midline_from_mask(lane_mask):
    """
    Tìm tập hợp tọa độ (x, y) nằm giữa hai biên lane trong ảnh mask sau birdview.
    Input:
        lane_mask: ảnh nhị phân (0-255), vùng lane đã đầy (không chỉ là edge)
    Output:
        midline_points: list[(x, y)] các tọa độ trung điểm từ dưới lên
    """

    # Đảm bảo ảnh 1 kênh
    if len(lane_mask.shape) == 3:
        lane_mask = cv2.cvtColor(lane_mask, cv2.COLOR_BGR2GRAY)

    # Chuẩn hóa về nhị phân
    _, binary = cv2.threshold(lane_mask, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    mid_points = []

    # Quét từng hàng ngang (y) từ dưới lên
    for y in range(h - 1, 0, -5):  # bước 5 pixel cho nhanh
        row = binary[y, :]
        x_coords = np.where(row > 0)[0]
        if len(x_coords) >= 2:
            x_mid = int((np.min(x_coords) + np.max(x_coords)) / 2)
            mid_points.append((x_mid, y))

    mid_points = np.array(mid_points)

    # Làm mượt nhẹ (fit đường cong)
    if len(mid_points) > 5:
        ys = mid_points[:, 1]
        xs = mid_points[:, 0]
        poly = np.polyfit(ys, xs, 2)  # x = a*y² + b*y + c
        ys_fit = np.linspace(min(ys), max(ys), 100)
        xs_fit = np.polyval(poly, ys_fit)
        mid_points = np.vstack((xs_fit, ys_fit)).T.astype(int)

    # Visualization
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (x, y) in mid_points:
        cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
    if len(mid_points) > 1:
        cv2.polylines(vis, [mid_points], False, (0, 255, 0), 2)
    cv2.imshow("Midline from Lane Mask", vis)
    cv2.waitKey(1)

    return mid_points

def visualize_midline(img, mid_points, color=(0, 255, 255)):
    """
    Hiển thị ảnh lane (mask hoặc edge) cùng với đường midline đã tính.
    Input:
        img: ảnh gốc hoặc mask sau birdview (1 hoặc 3 kênh)
        mid_points: list hoặc np.ndarray [(x, y), ...]
        color: màu vẽ midline (BGR)
    """

    # Đảm bảo ảnh 3 kênh để vẽ màu
    if len(img.shape) == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    # Vẽ midline
    if mid_points is not None and len(mid_points) > 1:
        pts = np.array(mid_points, dtype=np.int32)
        cv2.polylines(vis, [pts], False, color, 2)
        for (x, y) in pts[::10]:  # vẽ chấm mỗi 10 pixel cho dễ quan sát
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Midline Visualization", vis)
    cv2.waitKey(1)
    return vis