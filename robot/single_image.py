import pyrealsense2 as rs
import numpy as np
import cv2
def image(camera_type,save_path):
    '''
    The first camera type represents Scene graph camera
    The second camera type represesnts the Falcon Pick scene camera
    '''
    camera_list=["935322072092","919122073634"]
# Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(camera_list[int(camera_type)])
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        for i in range(100):

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

   
            
            cv2.imwrite(save_path,color_image)
   
    finally:

        # Stop streaming
        pipeline.stop()