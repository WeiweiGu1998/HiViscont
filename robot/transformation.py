import numpy as np
from scipy.spatial.transform import Rotation as R
def img_to_base(frame):
    x,y,z=frame
    xyz=np.array([[x],[y],[z],[1]])
    #Creating a Rotation Matrix from the calibrated quaternions
    r=R.from_quat([0.678941,0.669388,-0.243953,-0.177328])
    RotationMatrix=r.as_matrix()
    #Adding the translation matrix
    TranslationVector=np.asarray([[0.868671],[0.199274],[0.402014]],dtype=np.float32)
    #Creating the Transformation Matrix
    TransformationMatrix=np.hstack((RotationMatrix,TranslationVector))
    TransformationMatrix=np.vstack((TransformationMatrix,[0,0,0,1]))
    XYZ=TransformationMatrix@xyz
    #To make sure that it is pickible
    if XYZ[2]>30:
     
        XYZ[2]=25
    return XYZ
