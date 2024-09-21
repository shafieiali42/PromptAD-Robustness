import cv2 as cv

import cv2 as cv
import numpy as np
   
from PIL import Image
from imagecorruptions import corruppythot



# def gaussian_noise(x, severity=1):
#     c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


# def shot_noise(x, severity=1):
#     c = [60, 25, 12, 5, 3][severity - 1]

#     x = np.array(x) / 255.
#     return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


image=cv.imread("mvtec_anomaly_detection/carpet/train/good/000.png")

corrupted_image = corrupt(image, corruption_name='gaussian_blur', severity=1)
cv.imwrite("corrupted_using_library.jpg",corrupted_image)

print(image.shape)

# a=np.uint8(gaussian_noise(Image.fromarray(image),severity=5))
# print(a)
# cv.imwrite("corruped.jpg",a)