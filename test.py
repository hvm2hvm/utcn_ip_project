import cv2 as cv
import math
import numpy as np
import time
import sys

SEARCH_SIZE = 21
SEARCH_EDGE = 10
SAMPLE_SIZE = 7

img = cv.imread("/home/voicu/work/017_facultate/IP/Images/lab10/balloons_Gauss.bmp", cv.CV_8UC1)

coords = [[(x, y) for x in range(-3, 4)] for y in range(-3, 4)]
print(coords)

def euclidean(v, p):
    # result = v * math.sqrt(p[0]*p[0]+p[1]*p[1])
    # result = v / (1 + math.sqrt(p[0]*p[0] + p[1]*p[1]))
    result = v * v
    return result

f_euclidean = np.vectorize(euclidean, signature="(),(2)->()")

# print ("test: ", np.dot([1,2,3], [3,2,1]))
# sys.exit()

def get_similarity(img, p1, p2):
    slice1 = img[p1[0]-3:p1[0]+4, p1[1]-3:p1[1]+4].astype(np.int32)
    slice2 = img[p2[0]-3:p2[0]+4, p2[1]-3:p2[1]+4].astype(np.int32)
    # print("slice1 = ", slice1)
    # print("slcie1 square = ", np.square(slice1))
    # print("slice2 = ", slice2)
    slice1_norm = sum(sum(np.square(slice1)))
    slice2_norm = sum(sum(np.square(slice2)))
    dot_product = 2 * sum(sum(slice1 * slice2))
    # print ("s1 norm = ", slice1_norm)
    # print ("s2 norm = ", slice2_norm)
    # print ("dot = ", dot_product)
    result = slice1_norm + slice2_norm - dot_product
    # print ("result = ", result)
    # diff = slice1 - slice2
    # print("diff = ", diff)
    # diff = f_euclidean(diff, coords)
    # print("diff e = ", diff)
    # result = sum(sum(abs(diff)))
    return result

def get_constant(img, p1, h):
    sim = get_similarity(img, p1, p1)
    print ("sim = ", sim)
    return math.exp(-sim / h / h) * SEARCH_SIZE * SEARCH_SIZE

def get_weight(img, p1, p2, h):
    sim = get_similarity(img, p1, p2)
    # constant = get_constant(img, p1, h)
    constant = 1
    print("sim = ", sim)
    print("constant = ", constant)
    arg = -sim / (h*h)
    print("arg = ", arg)
    return math.exp(arg) / constant

def get_all_weights(img, p1, h):
    w = {}
    for dy in range(-SEARCH_EDGE, SEARCH_EDGE+1):
        for dx in range(-SEARCH_EDGE, SEARCH_EDGE+1):
            dpp = (dx, dy)
            pp = (p1[0]+dx, p1[1]+dy)
            w[dpp] = get_weight(img, p1, pp, h)
    return w

def show_rectangles(img, p1, p2):
    cv.rectangle(img, (p1[0]-3, p1[1]-3), (p1[0]+4, p1[1]+4), (255, 255, 255))
    cv.rectangle(img, (p2[0] - 3, p2[1] - 3), (p2[0] + 4, p2[1] + 4), (255, 255, 255))
    cv.imshow("sim location", img)

# print (get_similarity(img, (100, 240), (100, 260)))
# print (get_weight(img, (100, 240), (100, 260), 10))
# show_rectangles(img, (100, 240), (100, 260))
# cv.waitKey()
# time.sleep(100000)
weights = get_all_weights(img, (100, 240), 10)
print (weights)
print (sum(weights.values()))
for k, v in weights.items():
    if v > 1e-10:
        print("k, v: ", k, v)