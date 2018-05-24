
from Feature2D.Feature2DBase import *
import cv2

class SIFTFeature(Feature2DBase):

    def __init__(self):
        super(SIFTFeature, self).__init__()

    @classmethod
    def detectAndMatch(cls, image1, image2, mask1=None, mask2=None) -> (np.ndarray, np.ndarray):

        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d_SIFT.create()
        kp1, des1 = sift.detectAndCompute(image1_gray, mask1)
        kp2, des2 = sift.detectAndCompute(image2_gray, mask2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        good = []

        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        # ransac using homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)
        matchesmask = mask.ravel().tolist()

        pts1 = []
        pts2 = []
        for i in range(0, len(matchesmask) - 1):
            if matchesmask[i] == 1:
                pts1.append(src_pts[i])
                pts2.append(dst_pts[i])

        pts1 = np.float32(pts1).reshape(-1, 2)
        pts2 = np.float32(pts2).reshape(-1, 2)

        return pts1, pts2

if __name__ == "__main__":
    import os
    dir = "C:/Code/bird/AFGCD-master/data/data1"
    refImg = cv2.imread(os.path.join(dir, 'ref.png'))
    curImg = cv2.imread(os.path.join(dir, 'cur.png'))
    sift = SIFTFeature()
    ref_pts, dst_pts = sift.detectAndMatch(refImg,curImg)
    print(ref_pts.shape)
    print(dst_pts.shape)







