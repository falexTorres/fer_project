import numpy as np

images_in = np.load('../fer_X_train_smooth.npy')
labels_in = np.load('../fer_y_train.npy')
images_out = []
labels_out = []

zeros_out = []

ones_out = []

two_count = 0
twos_out = []

three_count = 0
threes_out = []

four_count = 0
fours_out = []

fives_out = []

six_count = 0
sixes_out = []

for i in range(0, images_in.shape[0]):
    if np.array_equal(labels_in[i], [0]):
      zeros_out.append(images_in[i])
    if np.array_equal(labels_in[i], [1]):
      ones_out.append(images_in[i])
    if np.array_equal(labels_in[i], [2]) and two_count < 4000:
      twos_out.append(images_in[i])
      two_count += 1
    if np.array_equal(labels_in[i], [3]) and three_count < 4000:
      threes_out.append(images_in[i])
      three_count += 1
    if np.array_equal(labels_in[i], [4]) and four_count < 4000:
      fours_out.append(images_in[i])
      four_count += 1
    if np.array_equal(labels_in[i], [5]):
      fives_out.append(images_in[i])
    if np.array_equal(labels_in[i], [6]) and six_count < 4000:
      sixes_out.append(images_in[i])
      six_count += 1

for i in range(0,5):
	zeros_out.append(zeros_out[i])

s = len(ones_out)

for j in range(0,8):
	for k in range(0,s):
		ones_out.append(ones_out[k])

for l in range(1,77):
		ones_out.append(ones_out[l])

for i in range(1,830):
	fives_out.append(fives_out[i])

for w in range(0, 7):
  for i in range(0, 4000):
    if w == 0:
    	labels_out.append([0])
    	images_out.append(zeros_out[i])
    if w == 1:
    	labels_out.append([1])
    	images_out.append(ones_out[i])
    if w == 2:
    	labels_out.append([2])
    	images_out.append(twos_out[i])
    if w == 3:
    	labels_out.append([3])
    	images_out.append(threes_out[i])
    if w == 4:
    	labels_out.append([4])
    	images_out.append(fours_out[i])
    if w == 5:
    	labels_out.append([5])
    	images_out.append(fives_out[i])
    if w == 6:
    	labels_out.append([6])
    	images_out.append(sixes_out[i])
    	
np.save('../fer_X_train_final.npy', images_out)
np.save('../fer_y_train_final.npy', labels_out)