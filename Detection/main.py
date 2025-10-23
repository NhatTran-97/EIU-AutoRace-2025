import cv2
import numpy as np 
import os 
# import config 
# from Lane.edge_segmentation import Segment_Edges
from Lane.colour_segmentation_final import Segment_Colour
# from Lane.colour_segmentation import Segment_Colour



def main():
    cap = cv2.VideoCapture(os.path.abspath("data/Lane_vid.avi"))
    while(1):
        ret, img = cap.read()
        if not ret:
            break
        minArea = 500
        waitTime = 0
        # Segment_Edges(img)

     #   Mid_edge_ROI, Mid_ROI_mask, Outer_edge_ROI, OuterLane_TwoSide, OuterLane_Points = Segment_Colour(img, minArea)

    
        Segment_Colour(img, minArea)

        cv2.imshow("Frame", img)
        k = cv2.waitKey(waitTime)
        if k == 27:
            break


if __name__ =='__main__':
    main()