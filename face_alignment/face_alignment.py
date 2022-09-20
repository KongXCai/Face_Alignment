from functools import total_ordering
import sys
sys.path.append('../')
import cv2
import os
import numpy as np
import mtcnn
from mtcnn.utils import draw
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank

#初始化人脸对齐前后的路径
source_dir = '/home/fxf/my_model/resnet50/resnet50_test/test_images'
save_dir=source_dir+"_aligned"
os.mkdir(save_dir)

#112*112的五点标准位置，如果是112*96，则左边一列的每个坐标减8
REFERENCE_FACIAL_POINTS = np.array([
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014 ],
    [70.72990036, 92.20410156]
], np.float32)

#使用最小二乘法进行仿射变换
def findNonreflectiveSimilarity(uv, xy, K=2):

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    T = T[:, 0:2].T

    return T

#获取权重
pnet, rnet, onet = mtcnn.get_net_caffe('../output/converted')

# 创建检测器
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')  #使用gpu时还需指定设备号:"cuda:0"     使用cpu:"cpu"

#特征提取
dir_class = next(os.walk(source_dir))[1]
img_class = next(os.walk(source_dir))[2]
dir_class.sort()
img_class.sort()

#对目录下的子目录照片进行人脸对齐
print("正在进行对齐运算中……")
success_num=0
fail_num=0
for dir in dir_class:
    index=1
    img_dir=os.path.join(source_dir,dir)
    img_dir_aligned=os.path.join(save_dir,dir)
    os.mkdir(img_dir_aligned)
    for img in os.listdir(img_dir):
        format=img.split(".")[1]
        img_rename="/"+dir+"_"+str(index)+"."+format
        img = cv2.imread(os.path.join(img_dir,img))
        boxes, landmarks = detector.detect(img, minsize=24)
        if(len(landmarks)==0):
            print(img_rename,"对齐运算失败")
            fail_num+=1
            continue
        landmark=landmarks[0]
        similar_trans_matrix = findNonreflectiveSimilarity(landmark.cuda().numpy().astype(np.float32), REFERENCE_FACIAL_POINTS)
        aligned_img = cv2.warpAffine(img.copy(), similar_trans_matrix, (112, 112)) 
        cv2.imwrite(img_dir_aligned+img_rename, aligned_img)
        print("文件已成功保存到",img_dir_aligned+img_rename)
        index+=1
    success_num=success_num+index

#对目录下的照片文件进行人脸对齐
for img in img_class:
    img_rename="/"+img
    img = cv2.imread(os.path.join(source_dir,img))
    boxes, landmarks = detector.detect(img, minsize=24)
    landmark=landmarks[0]
    similar_trans_matrix = findNonreflectiveSimilarity(landmark.cpu().numpy().astype(np.float32), REFERENCE_FACIAL_POINTS)
    aligned_img = cv2.warpAffine(img.copy(), similar_trans_matrix, (112, 112))
    cv2.imwrite(save_dir+img_rename, aligned_img)
    print("文件已成功保存到",save_dir+img_rename)
    success_num=success_num+1

print("人脸对齐运算完毕！")
print("成功:",success_num)
print("失败:",fail_num)