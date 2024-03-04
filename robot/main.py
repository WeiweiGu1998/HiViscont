from Vison import *
import cv2
from scipy import ndimage
from single_image import image
import json

"""
0: Falcon Pick Scene 
1: Scengraph Pick Scene
"""
def create_json_and_image(image_path,annotation_path,camera_no,json_type):
        
    # img_path=path+"/"+name+".jpg"
    # json_path=path+"/"+name+".json"
    image(camera_no,image_path)
    img=cv2.imread(image_path)
    sam=sam_initiazliation()
    mask_generator=sam.sam_setup()
    seg=segment(img,mask_generator)
    mask=seg.get_mask()
    centroid=mask_centroid(mask)
    masks,bboxes=centroid.main()
    d={}
    d["bboxes"]=bboxes
    if json_type==0:
        t1="green square tiles are used to create the floor"
        t2="red rectangular short tiles are used to make the pillar "
        t3="blue curved block are used to make the roof"
        t0="ground"
        
        d["text"]=[t0,t1,t2,t3]
        d["segment"]=[0,4,6,8]
    
    if json_type==1:
        tc="This is a yellow square tile it has the property of flooring and color of blue"
        d["text"]=tc
    

    with open(annotation_path, 'w') as fp:
        json.dump(d, fp)

PICK_SCENE_IMAGE_PATH = "/home/local/ASUAD/asah4/projects/test/annotations/FALCON-Generalized/robot/pick_scene.jpg"
# PICK_SCENE_ANNOTATION_PATH = "/home/local/ASUAD/asah4/projects/test/annotations/FALCON-Generalized/robot/pick_scene_demo.json"
image=cv2.imread(PICK_SCENE_IMAGE_PATH)
x,y,w,h=[406, 275, 59, 57]
cv2.rectangle(image,(x,y),((x+w),(y+h)),(255,0,0), 2)
cv2.imshow("Checker",image)
cv2.waitKey(0)
# create_json_and_image(PICK_SCENE_IMAGE_PATH,PICK_SCENE_ANNOTATION_PATH,0,2)


 