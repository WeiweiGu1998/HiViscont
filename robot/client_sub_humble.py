import rclpy
from sensor_msgs.msg import PointCloud2
import numpy as np
from transformation import img_to_base
class DepthListner:
    def __init__(self,frame):
        self.camera=None
        self.frame=frame
        rclpy.init()
        self.node = rclpy.create_node('calib_node')
        self.subcription=self.node.create_subscription(PointCloud2, '/camera/depth/color/points',self.callback, 10)
        self.subcription

    def callback(self,msg):
        np_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('b0', '<f4'), ('rgb', '<f4')]
        np_pc = np.frombuffer(msg.data, dtype=np_dtype)
        points = np.expand_dims(np.hstack((np.expand_dims(np_pc['x'],-1), np.expand_dims(np_pc['y'], -1), np.expand_dims(np_pc['z'],-1))), 0)
        points = points.reshape((480,640,3))
        rgb = np.frombuffer(np.ascontiguousarray(np_pc['rgb']).data, dtype=np.uint8)
        rgb = np.expand_dims(rgb,0).reshape(480*640,4)[:,:3]
        rgb = np.expand_dims(rgb,0).reshape(480,640,3)
        self.camera=points[self.frame[1],self.frame[0]]
    def camera_frame(self):
        return self.camera
    
def base_frame(frame):
    d=DepthListner(frame)
    rclpy.spin_once(d.node)
    rclpy.shutdown()
    x,y,z=d.camera_frame()
    XYZ=img_to_base([x,y,z])
    return XYZ

# if __name__=='__main__':
#   print( main([224,67]))