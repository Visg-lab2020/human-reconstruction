import numpy as np
import math
import os
class Node:
    def __init__(self, data, lchild = None, rchild = None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild

class KdTree:
    def __init__(self):
        self.kdTree = None

    def create(self, dataSet, depth):
        if (len(dataSet) > 0):
            m, n = np.shape(dataSet)
            lvl=2
            midIndex = int(m / 2)
            axis = depth % lvl
            sortedDataSet = self.sort(dataSet, axis)
            node = Node(sortedDataSet[midIndex])
            # print sortedDataSet[midIndex]
            leftDataSet = sortedDataSet[: midIndex]
            rightDataSet = sortedDataSet[midIndex+1 :]
            #print(leftDataSet)
            #print(rightDataSet)
            node.lchild = self.create(leftDataSet, depth+1)
            node.rchild = self.create(rightDataSet, depth+1)
            return node
        else:
            return None

    def sort(self, dataSet, axis):
        sortDataSet = dataSet[:]
        m, n = np.shape(sortDataSet)
        
        for i in range(m):
            for j in range(0, m - i - 1):
                if (sortDataSet[j][axis] > sortDataSet[j+1][axis]):
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j+1]
                    sortDataSet[j+1] = temp
        #print(sortDataSet)
        return sortDataSet

    def preOrder(self, node):
        if node != None:
            #print("tttt->%s" % node.data)
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)


    def search(self, tree, x):
        self.nearestPoint = None
        self.nearestValue = 0
        def travel(node, depth = 0):
            if node != None:
                n = len(x)
                axis = depth % 2
                if x[axis] < node.data[axis]:
                    travel(node.lchild, depth+1)
                else:
                    travel(node.rchild, depth+1)

                
                distNodeAndX = self.dist(x, node.data)
                if (self.nearestPoint == None):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX

                #print(node.data, depth, self.nearestValue, node.data[axis], x[axis])
                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):
                    if x[axis] < node.data[axis]:
                        travel(node.rchild, depth+1)
                    else:
                        travel(node.lchild, depth + 1)
        travel(tree)
        return self.nearestPoint

    def dist(self, x1, x2):
        return ((np.array(x1[:2]) - np.array(x2[:2])) ** 2).sum() ** 0.5


def gen_list(IUV_left,IUV_right):
    UV_list1=[0]*24
    UV_list2=[0]*24
    
    for i in range(0,24):
        UV_list1[i]=[]
        UV_list2[i]=[]
    for m in range(0,IUV_left.shape[0]):
        for n in range(0, IUV_left.shape[1]):
            label1=IUV_left[m, n,0]
            u1 = IUV_left[m, n,1]
            v1 = IUV_left[m, n,2]
            label2=IUV_right[m, n,0]
            u2 = IUV_right[m, n,1]
            v2 = IUV_right[m, n,2]
           
            if (label1!=0):
                label1=int(label1-1)
                UV_list1[label1].append([u1,v1,m,n])
	    if (label2!=0):
                label2=int(label2-1)
                UV_list2[label2].append([u2,v2,m,n])
    return UV_list1,UV_list2


for pair in range(0,4000):
    str='%06d'%pair
    print(pair)
    if os.path.exists('//data2/hc/0-25-50/UV_0/'+str+'_UV.npy'):
        if os.path.exists('//data2/hc/0-25-50/UV_2/'+str+'_UV.npy'):
            IUV_left=np.load('//data2/hc/0-25-50/UV_0/'+str+'_UV.npy')
            IUV_right=np.load('//data2/hc/0-25-50/UV_2/'+str+'_UV.npy')
            UV_list1,UV_list2=gen_list(IUV_left,IUV_right)
            points=[]
            for i in range(0,24):
                k1=len(UV_list1[i])
                k2=len(UV_list2[i])
                if (k1!=0 and k2!=0 ):
                    kdtree = KdTree()
                    tree=kdtree.create(UV_list2[i], 0)
                    kdtree.preOrder(tree)           
                    for j in range(0,k1):
                        uv1=UV_list1[i][j]
                        uv2=kdtree.search(tree, uv1)
                        u1_gt=uv1[0]
                        v1_gt=uv1[1]
                        u2_gt=uv2[0]
                        v2_gt=uv2[1]
                        loss1=math.sqrt((u1_gt-u2_gt)** 2+(v1_gt-v2_gt)** 2)
                        if (loss1<0.1):
                            x1=uv1[2]
                            y1=uv1[3]
                            x2=uv2[2]
                            y2=uv2[3]
                            point=[x1,y1,x2,y2]
                            points.append(point)
            np.save('//data2/hc/0-25-50/corre_02/'+str+'.npy',points)
            print(len(points))
           
            
        








