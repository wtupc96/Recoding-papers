import sys

sys.path.append('/home/se-rhr/oo/recoding_papers')

from Visualizing_and_Understanding_Convolutional_Networks.conv import train
from Visualizing_and_Understanding_Convolutional_Networks.deconv import run

input_image, image_conv1, image_conv2, image_conv3 = train()
restruct_conv1, restruct_conv2, restruct_conv3 = run(image_conv1, image_conv2, image_conv3)

# fig = plt.figure()
# ax1 = fig.add_subplot(1, 4, 1)
# ax1.plot(input_image)
#
# # ax2 = fig.add_subplot(1, 7, 2)
# # ax2.plot(image_conv1)
# #
# # ax3 = fig.add_subplot(1, 7, 3)
# # ax3.plot(image_conv2)
# #
# # ax4 = fig.add_subplot(1, 7, 4)
# # ax4.plot(image_conv3)
#
# ax5 = fig.add_subplot(1, 4, 2)
# ax5.plot(restruct_conv1)
#
# ax6 = fig.add_subplot(1, 4, 3)
# ax6.plot(restruct_conv2)
#
# ax7 = fig.add_subplot(1, 4, 4)
# ax7.plot(restruct_conv3)
#
# fig.savefig('full_figure.png')

import cv2
print(input_image.shape)
cv2.imwrite('1.png', input_image)
print(image_conv1.shape)
cv2.imwrite('2.png', restruct_conv1)
print(image_conv2.shape)
cv2.imwrite('3.png', restruct_conv2)
print(image_conv3.shape)
cv2.imwrite('4.png', restruct_conv3)