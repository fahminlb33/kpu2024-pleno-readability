import cv2
import numpy as np

img_template = cv2.imread("dataset/1101102001002_002_p1.jpg")
img_query = cv2.imread("dataset/1101062028003_003_p1.jpg")

orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img_template, None)
kp2, des2 = orb.detectAndCompute(img_query, None)

# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)

good_matches = [m.distance < 64 for m in matches]

print("Good matches:", sum(good_matches))
print("Similarity:", sum(good_matches) / len(matches))
print("Similarity raw:", len(matches) / len(kp1))
print([x.distance for x in matches])

# cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatches(img_template, kp1, img_query, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv2.imshow("SIFT", img3)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
