import numpy as np
import math
import cv2
import os


TAG_FLOAT=202021.25

def clean_dst_file(dst_file):
    """Create the output folder, if necessary; empty the output folder of previous predictions, if any
    Args:
        dst_file: Destination path
    """
    # Create the output folder, if necessary
    dst_file_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_dir):
        os.makedirs(dst_file_dir)

    # Empty the output folder of previous predictions, if any
    if os.path.exists(dst_file):
        os.remove(dst_file)




def flow_write(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

'''
left image flow(to right)
'''

'''
for pair in range(0,10):
    print(pair)
    str='%06d'%pair
    points=np.load('//data2/hc/MVFR/clean_35-50/val/corre_left/'+str+'.npy')
    img=readFlow('//data2/hc/MVFR/clean_35-50/val/flow_gt/'+str+'BA_gt.flo')
    
    IUV=np.load('//data2/hc/MVFR/clean_35-50/val/UV_left/'+str+'_UV.npy')
    #print(IUV.shape)
    flow=np.zeros((256,256,2))
    mask=np.zeros((256,256,3))
    for m in range(0,img.shape[0]):
        for n in range(0,img.shape[1]):
            if (img[m,n,0]==-10):
                mask[m,n,0]=-255
                mask[m,n,1]=-255
                mask[m,n,2]=-255
    mask=mask+255

    for i in range(0,points.shape[0]):
        x1=points[i,0]
        y1=points[i,1]
        x2=points[i,2]
        y2=points[i,3]
        x1_int=int(x1) 
        y1_int=int(y1) 
        flow[x1_int,y1_int,0]=y2-y1
        flow[x1_int,y1_int,1]=x2-x1
    for m in range(0,img.shape[0]):
        for n in range(0,img.shape[1]):
            if (img[m,n,0]==-10 and img[m,n,1]==-10):
                flow[m,n,0]=-255
                flow[m,n,1]=-255
    flow=flow*2./255
    #np.save('//data2/hc/MVFR/clean_surreal/val/flow_dense/'+str+'_BA.npy',flow)
    cv2.imwrite('//data2/hc/MVFR/clean_35-50/val/mask/'+str+'_left.jpg',mask)
    flow_write(flow,'//data2/hc/MVFR/clean_35-50/val/flow_dense_flo/'+str+'_BA.flo')
'''

'''
right image flow(to left)
'''


for pair in range(0,10):
    print(pair)
    str='%06d'%pair
    points=np.load('//data2/hc/MVFR/clean_35-50/val/corre_right/'+str+'.npy')
    IUV=np.load('//data2/hc/MVFR/clean_35-50/val/UV_right/'+str+'_UV.npy')
    img=readFlow('//data2/hc/MVFR/clean_35-50/val/flow_gt/'+str+'AB_gt.flo')
    #img=img[:,256:,:]
    #print(IUV.shape)
    flow=np.zeros((256,256,2))
    mask=np.zeros((256,256,3))
    for m in range(0,img.shape[0]):
        for n in range(0,img.shape[1]):
            if (img[m,n,0]==-10):
                mask[m,n,0]=-255
                mask[m,n,1]=-255
                mask[m,n,2]=-255
    mask=mask+255

    for i in range(0,points.shape[0]):
        x1=points[i,0]
        y1=points[i,1]
        x2=points[i,2]
        y2=points[i,3]
        x2_int=int(x2) 
        y2_int=int(y2) 
        flow[x2_int,y2_int,0]=y1-y2
        flow[x2_int,y2_int,1]=x1-x2
    for m in range(0,img.shape[0]):
        for n in range(0,img.shape[1]):
            if (img[m,n,0]==-10 and img[m,n,1]==-10):
                flow[m,n,0]=-255
                flow[m,n,1]=-255
    flow=flow*2./255
     
    #np.save('//data2/hc/MVFR/clean_surreal/val/flow_dense/'+str+'_AB.npy',flow)
    cv2.imwrite('//data2/hc/MVFR/clean_35-50/val/mask/'+str+'_right.jpg',mask)
    flow_write(flow,'//data2/hc/MVFR/clean_35-50/val/flow_dense_flo/'+str+'_AB.flo')

