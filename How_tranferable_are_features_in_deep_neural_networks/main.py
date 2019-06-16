import sys

sys.path.append('/home/se-rhr/oo/recoding_papers')

from Visualizing_and_Understanding_Convolutional_Networks.conv import train
from Visualizing_and_Understanding_Convolutional_Networks.deconv import run

input_image, image_conv1, image_conv2, image_conv3 = train()
restruct_conv1, restruct_conv2, restruct_conv3 = run(image_conv1, image_conv2, image_conv3)

import cv2

print(input_image.shape)
cv2.imwrite('1.png', input_image)
print(image_conv1.shape)
cv2.imwrite('2.png', restruct_conv1)
print(image_conv2.shape)
cv2.imwrite('3.png', restruct_conv2)
print(image_conv3.shape)
cv2.imwrite('4.png', restruct_conv3)
