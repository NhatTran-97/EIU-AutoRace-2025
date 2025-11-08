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

from Lane.c_Cleaning.CheckifYellowLaneCorrect_RetInnerBoundary import GetYellowInnerEdge
from Lane.c_Cleaning.ExtendLanesAndRefineMidLaneEdge import ExtendShortLane

from Lane.d_LaneInfo_Extraction.GetStateInfoandDisplayLane import FetchInfoAndDisplay


Use_Threading = False 
Live_Testing = False 

# if Use_Threading:
#     from imutils.video.pivideostream import piVideoStream


def main():
    cap = cv2.VideoCapture(os.path.abspath("data/Lane_vid.avi"))
    waitTime = 0


    while(True):
        ret, img = cap.read()
        if not ret:
            break
        
        img = cv2.resize(img, (320, 240))
        # CropHeight = 260 
        # minArea = 500 

        CropHeight = 130
        minArea = 250
        img = img[CropHeight:, :]
        
        
        # Segment_Edges(img)

        # ********************************************************** DETECTION *********************************************************

        # [Lane Detection] STAGE 1 (Segmentation)

        Mid_edge_ROI, Mid_ROI_mask, Outer_edge_ROI, OuterLane_TwoSide, OuterLane_Points = Segment_Colour(img, minArea)

        print("img shape:", img.shape, img.dtype)
        print("Mid_ROI_mask shape:", Mid_ROI_mask.shape, Mid_ROI_mask.dtype)

        Mid_ROI_mask = cv2.resize(Mid_ROI_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("OuterLane_TwoSide;;;;", OuterLane_TwoSide)
        # [Lane Detection] STAGE_2 (Estimation) <<<<<----->>>> [ Our Approach]
        Midlane_segmented_Rgb = cv2.bitwise_and(img, img, mask=Mid_ROI_mask)
        # cv2.imshow('[Midlane_segmented_Rgb]', Midlane_segmented_Rgb)

        # Hough(Midlane_segmented_Rgb)
        Estimated_midlane = Estimate_MidLane(Mid_edge_ROI, config.MaxDist_resized)
        cv2.imshow("Estimated_midlane", Estimated_midlane)
      

        # [Lane Detection] STAGE_3 (Cleaning) <<<--->>> [STEP 1]:

        # [Lane Detection] STAGE_3 (Cleaning) <<<<<<--->>>>>> [STEP_1]:
        OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction = GetYellowInnerEdge(OuterLane_TwoSide,Estimated_midlane,OuterLane_Points)#3ms
   
        # [Lane Detection] STAGE_3 (Cleaning) <<<<<<--->>>>>> [STEP_2]:
        Estimated_midlane,OuterLane_OneSide = ExtendShortLane(Estimated_midlane,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)
        cv2.imshow("Estimated_midlane:::::", Estimated_midlane)
        cv2.imshow("OuterLane_OneSide:::::", OuterLane_OneSide)

        Distance, Curvature = FetchInfoAndDisplay(Mid_edge_ROI, Estimated_midlane, Outer_cnts_oneSide,img, Offset_correction )

        
        

    
        # Segment_Colour(img, minArea)

        cv2.imshow("Frame", img)
        k = cv2.waitKey(waitTime)
        if k == 27:
            break


if __name__ =='__main__':
    main()




#  python3 -m cProfile -s tottime main.py,  python3 -m cProfile -s cumtime main.py