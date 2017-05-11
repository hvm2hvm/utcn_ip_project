import cv2 as cv
import math
import numpy as np
import time
import sys

SEARCH_SIZE = 21
SEARCH_EDGE = SEARCH_SIZE // 2
SAMPLE_SIZE = 7
SAMPLE_EDGE = SAMPLE_SIZE // 2

img = cv.imread("/home/voicu/work/017_facultate/IP/Images/lab10/balloons_Gauss.bmp", cv.CV_8UC1)

def gaussian_element(u, v):
    return math.exp

sigma = 2.0
gauss = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), np.float)

for i in range(SAMPLE_SIZE):
    for j in range(SAMPLE_SIZE):
        di = i - SAMPLE_EDGE
        dj = j - SAMPLE_EDGE
        value = math.exp(-(di * di + dj * dj) / (2 * sigma * sigma))
        # print (di, dj, value)
        gauss[i][j] = value

gauss_sum = sum(sum(gauss))

def get_weight(img, p1, p2):
    diff_sum = 0
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    for dx in range(-SAMPLE_EDGE, SAMPLE_EDGE+1):
        for dy in range(-SAMPLE_EDGE, SAMPLE_EDGE+1):
            xx1, yy1 = x1 + dx, y1 + dy
            xx2, yy2 = x2 + dx, y2 + dy

            diff_sum += gauss[dx+SAMPLE_EDGE][dy+SAMPLE_EDGE] * (int(img[yy1][xx1]) - int(img[yy2][xx2])) ** 2

    return math.exp(-diff_sum / gauss_sum)

def denoise_pixel(img, p1):
    weighted_sum = 0.0
    w_sum = 0.0
    for dx in range(-SEARCH_EDGE, SEARCH_EDGE+1):
        for dy in range(-SEARCH_EDGE, SEARCH_EDGE+1):
            xx, yy = p1[0] + dx, p1[1] + dy
            p2 = p1[0] + dx, p1[1] + dy
            w = get_weight(img, p1, p2)
            weighted_sum += w * img[yy][xx]
            w_sum += w

    # print("weighted sum: ", weighted_sum)
    # print("w sum: ", w_sum)

    return weighted_sum / w_sum

def denoise_image(img):
    dest = img.copy()

    diff_sum = 0.0
    x1, x2 = 100, 130
    y1, y2 = 200, 230
    for x in range(x1, x2+1):
        print("x = ", x)
        for y in range(y1, y2+1):
            dest[y][x] = denoise_pixel(img, (x, y))
            diff_sum += abs(dest[y][x] - img[y][x])
    print("diff sum = ", diff_sum)
    cv.rectangle(dest, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255))

    return dest

cv.imshow("source", img)
cv.imshow("dest", denoise_image(img))
cv.waitKey()
time.sleep(100000)
