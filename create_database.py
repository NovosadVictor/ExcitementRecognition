import pickle
from PIL import Image
import random
import numpy as np


all_data = []
size_of_test = 0.1

for i in range(1, 21):
    # 6 timesteps (6 images(3 sec) per one result)
    images = []
    for j in range(6):
        image = Image.open('./videos_for_project/exam_1/images%06d.png' % (i + j))
        pix = image.load()
        img_matrix = [(pix[l,k][0] + pix[l,k][1] + pix[l,k][2]) // 3 for k in range(180)
                      for l in range(200)]
        images.append(img_matrix)
    print("step " + str(i) + " of " + str(20))
    all_data.append([images, [1, 0]])
print("Exam 1 Done")

for i in range(1, 21):
    # 6 timesteps (6 images(3 sec) per one result)
    images = []
    for j in range(6):
        image = Image.open('./videos_for_project/exam_2/images%06d.png' % (i + j))
        pix = image.load()
        img_matrix = [(pix[l,k][0] + pix[l,k][1] + pix[l,k][2]) // 3 for k in range(180)
                      for l in range(200)]
        images.append(img_matrix)

    print("step " + str(i) + " of " + str(20))
    all_data.append([images, [1, 0]])
print("Exam 2 Done")

for i in range(1, 11):
    # 6 timesteps (6 images(3 sec) per one result)
    images = []
    for j in range(6):
        image = Image.open('./videos_for_project/not_1/images%06d.png' % (i + j))
        pix = image.load()
        img_matrix = [(pix[l,k][0] + pix[l,k][1] + pix[l,k][2]) // 3 for k in range(180)
                      for l in range(200)]
        images.append(img_matrix)

    print("step " + str(i) + " of " + str(10))
    all_data.append([images, [0, 1]])
print("Not 1 Done")

for i in range(1, 11):
    # 6 timesteps (6 images(3 sec) per one result)
    images = []
    for j in range(6):
        image = Image.open('./videos_for_project/not_2/images%06d.png' % (i + j))
        pix = image.load()
        img_matrix = [(pix[l,k][0] + pix[l,k][1] + pix[l,k][2]) // 3 for k in range(180)
                      for l in range(200)]
        images.append(img_matrix)

    print("step " + str(i) + " of " + str(10))
    all_data.append([images, [0, 1]])
print("Not 2 Done")

for i in range(1, 11):
    # 6 timesteps (6 images(3 sec) per one result)
    images = []
    for j in range(6):
        image = Image.open('./videos_for_project/not_3/images%06d.png' % (i + j))
        pix = image.load()
        img_matrix = [(pix[l,k][0] + pix[l,k][1] + pix[l,k][2]) // 3 for k in range(180)
                      for l in range(200)]
        images.append(img_matrix)

    print("step " + str(i) + " of " + str(10))
    all_data.append([images, [0, 1]])
print("Not 3 Done")

for i in range(1, 11):
    # 6 timesteps (6 images(3 sec) per one result)
    images = []
    for j in range(6):
        image = Image.open('./videos_for_project/not_4/images%06d.png' % (i + j))
        pix = image.load()
        img_matrix = [(pix[l,k][0] + pix[l,k][1] + pix[l,k][2]) // 3 for k in range(180)
                      for l in range(200)]
        images.append(img_matrix)

    print("step " + str(i) + " of " + str(10))
    all_data.append([images, [0, 1]])
print("Not 4 Done")

print(len(all_data))

random.shuffle(all_data)

all_data = np.array(all_data)

images_data = list(all_data[:, 0])
labels_data = list(all_data[:, 1])

test_size = int(size_of_test * len(all_data))

train_x = images_data[:-test_size]
train_y = labels_data[:-test_size]

test_x = images_data[-test_size:]
test_y = labels_data[-test_size:]

if __name__ == '__main__':
    with open('database.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)




