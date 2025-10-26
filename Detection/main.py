import cv2
import numpy as np 
import os 
# import config 
# from Lane.edge_segmentation import Segment_Edges
from Lane.colour_segmentation_final import Segment_Colour
from Lane.b_Hough_Line_Estimation.HoughLineTransform import Hough
from Lane.b_Hough_Line_Estimation.Our_EstimationAlgorithm import Estimate_MidLane
from Lane import config
# from Lane.colour_segmentation import Segment_Colour



def main():
    cap = cv2.VideoCapture(os.path.abspath("data/Lane_vid.avi"))
    waitTime = 0


    while(1):
        ret, img = cap.read()
        if not ret:
            break

        CropHeight = 260 
        img_cropped = img[CropHeight:, :]
        minArea = 500
        
        # Segment_Edges(img)

        # ********************************************************** DETECTION *********************************************************

        # [Lane Detection] STAGE 1 (Segmentation)

        Mid_edge_ROI, Mid_ROI_mask, Outer_edge_ROI, OuterLane_TwoSide, OuterLane_Points = Segment_Colour(img, minArea)

        # print("img shape:", img.shape, img.dtype)
        # print("Mid_ROI_mask shape:", Mid_ROI_mask.shape, Mid_ROI_mask.dtype)

        # Mid_ROI_mask = cv2.resize(Mid_ROI_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


        # [Lane Detection] STAGE_2 (Estimation) <<<<<----->>>> [Our Approach]
        Midlane_segmented_Rgb = cv2.bitwise_and(img, img, mask=Mid_ROI_mask)
        cv2.imshow('[Midlane_segmented_Rgb]', Midlane_segmented_Rgb)

        # Hough(Midlane_segmented_Rgb)
        Estimated_midlane = Estimate_MidLane(Mid_edge_ROI, config.MaxDist_resized)
        cv2.imshow("Estimated_midlane", Estimated_midlane)

    
        # Segment_Colour(img, minArea)

        cv2.imshow("Frame", img)
        k = cv2.waitKey(waitTime)
        if k == 27:
            break


if __name__ =='__main__':
    main()