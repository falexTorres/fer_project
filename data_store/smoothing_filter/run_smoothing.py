import cv2
import numpy as np
from PIL import Image
import sys

SIZE_FACE = 48

def format_image(image): 
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  
  image = cv2.blur(image, (9,9))
  image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  return image

images_in = np.load('../fer_X_train_fdt.npy')
images_out = []

for i in range(0, images_in.shape[0]):
    #img = Image.fromarray(images_in[i]).convert('L')
    #img = np.array(img)[:, :].copy()
    #tmp = format_image(img)
    images_out.append(cv2.GaussianBlur(images_in[i], (5, 5), 0))

print "Total: " + str(len(images_out))
np.save('../fer_X_train_smooth.npy', images_out)
