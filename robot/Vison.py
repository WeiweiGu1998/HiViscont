from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import imutils
from skimage.morphology import label

class sam_initiazliation:
    def __init__(self):
        self.sam_checkpoint = "/home/local/ASUAD/asah4/projects/test/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda"
    def sam_setup(self):
        
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=11,
            pred_iou_thresh=0.98,
            stability_score_thresh=0.95,
            crop_n_layers=2,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=125,  # Requires open-cv to run post-processing
        )
        return mask_generator

        
class segment:
    def __init__(self,image,mask_generator):
        self.image=image
        self.mask_generator=mask_generator
    def get_mask(self):
        masks = self.mask_generator.generate(self.image)
        masks = [mask['segmentation'] 
                 for mask in sorted(masks, key=lambda x: x['area'], reverse=True)]    
        mask=np.array(masks[0],np.uint8)
        # mask=np.expand_dims(mask,axis=0)
        mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        return mask
        
class mask_centroid:
    def __init__(self,mask):
        self.mask=mask
    def preprocessing(self):
        img=np.invert(self.mask)
        x,y,z=img.shape
        img=img[4:x-4, 4:y-4,:]
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def erosion(self,img):
        #Adding erosion to the image
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        ret, th1 = cv2.threshold(v,180,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((1,1), dtype = "uint8")/9
        bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
        erosion_image = cv2.erode(bilateral, kernel, iterations = 1)
        return erosion_image
    
    def countours(self,erosion,img):
        #Adding Countours to the image
        pixel_components, output, stats, centroids =cv2.connectedComponentsWithStats(erosion, connectivity=8)
        area = stats[1:, -1]; pixel_components = pixel_components - 1
        min_size = 1000#your answer image
        img2 = np.zeros((img.shape))#Removing the small white pixel area below the minimum size
        for i in range(0, pixel_components):
            if area[i] >= min_size:
                img2[output == i + 1] = 255        
        img3 = img2.astype(np.uint8)       
        print(img3.shape)
        # find contours in the thresholded image
        img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(img4.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # return cnts,img3
        bbox=[]
        for j in cnts:
            rect = cv2.boundingRect(j)
            # x,y,w,h = rect
            bbox.append(rect)
            # cv2.rectangle(img4, (x,y),(x+w,y+h),(255,0,255),2)
            # plt.imshow(img4)

    
        img3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        labeled_mask = label(img3)
        print ('Found ', len(np.unique(labeled_mask)), ' connected masks')
        B = []
        for i in np.unique(labeled_mask):
            if i == 0: # id = 0 is for background
                continue
            mask_i = (labeled_mask==i).astype(np.uint8)
            B.append(mask_i)
        return B,bbox
 

    def main(self):

        img=self.preprocessing()
        erosion_image=self.erosion(img)
        B,bbox=self.countours(erosion_image,img)
        
        # B=self.mask(image)
        return B,bbox
    