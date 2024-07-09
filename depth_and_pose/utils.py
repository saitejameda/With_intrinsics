import numpy as np
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d
import cv2
import torch

def compute_matches(image0,image1,extractor,matcher):
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    return m_kpts0.cpu(),m_kpts1.cpu()

def comute_points_shift(img1,img2):
    sift = cv2.SIFT_create()
    img1 = img1.permute(1,2,0)
    img2 = img2.permute(1,2,0)
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    
    #if img1.dtype != np.uint8:
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    print('img1_type',type(img1),img1.shape)
    print('img2_type',type(img2),img2.shape)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)  

    #print("img1_type(img1): ",type(img1))
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return np.array(pts1),np.array(pts2)
    
def compute_Essential_matrix(image0, image1,extractor,matcher,K):
    print('utils_image',image0.shape,'image1',image1.shape)
    m_kpts0, m_kpts1 = compute_matches(image0, image1,extractor,matcher)
    if m_kpts0.shape[0]<8 or m_kpts1.shape[0]< 8 or m_kpts0 == None or m_kpts1 == None:
        m_kpts0,m_kpts1 = comute_points_shift(image0, image1)
    F,mask = cv2.findFundamentalMat(np.array(m_kpts0), np.array(m_kpts1),  cv2.FM_8POINT)
    if F is None:
        print(F,m_kpts1,m_kpts0)
    #print(type(F),type(K))
    E = np.dot(np.dot(K.detach().cpu().numpy().T, F), K.detach().cpu().numpy())
    #print(type(E),type(mask))
    return m_kpts0, m_kpts1, E ,mask

def compute_camera_pose(image0,image1,extractor,matcher,K):
    print('utils_image',image0.shape)
    pose_list = []
    for i in range(image0.shape[0]):
        m_kpts0, m_kpts1, E ,mask = compute_Essential_matrix(image0[i],image1[i],extractor,matcher,K[i])
        de_compose= cv2.recoverPose(E=E,points1= np.array(m_kpts0),
                                    points2= np.array(m_kpts1), cameraMatrix=K[i].detach().cpu().numpy(), distanceThresh=1.0, mask=mask)
        R = de_compose[1]
        t = de_compose[2]
        cam_pose = np.hstack((R, t))
        #cam_pose = np.vstack((cam_pose, np.array([0, 0, 0, 1])))
        pose_list.append(cam_pose)
    return torch.tensor(pose_list)
