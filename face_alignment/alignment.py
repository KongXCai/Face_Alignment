import sys
sys.path.append('../')
import cv2
import numpy as np
import mtcnn
from mtcnn.utils import draw
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as rank

img_file = './imge_example/before_aligned.jpg'
REFERENCE_FACIAL_POINTS = np.array([
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56.02519989, 71.73660278],
    [41.54930115, 92.3655014 ],
    [70.72990036, 92.20410156]
], np.float32)

#获取权重
pnet, rnet, onet = mtcnn.get_net_caffe('../output/converted')

# 创建检测器
detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')  #使用gpu时还需指定设备号:"cuda:0"     使用cpu:"cpu"

#特征提取
img = cv2.imread(img_file)
boxes, landmarks = detector.detect(img, minsize=24)
landmark=landmarks[0]


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

similar_trans_matrix = findNonreflectiveSimilarity(landmark.cpu().numpy().astype(np.float32), REFERENCE_FACIAL_POINTS)
aligned_face = cv2.warpAffine(img.copy(), similar_trans_matrix, (112, 112))

cv2.imwrite("./imge_example/after_aligned.jpg", aligned_face)
print("人脸对齐完毕！")