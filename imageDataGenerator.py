import numpy as np
import string
import glob
from pathlib import Path
import os

# 랜덤시드 고정시키기
np.random.seed(5)

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 데이터셋 불러오기
data_aug_gen = ImageDataGenerator(rescale=None,
                                  rotation_range=15,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                  #vertical_flip=True,
                                  fill_mode='nearest')

directory = ['m11', 'm12']
#for i in range(15):
#    directory.append('j' + str(i+1))

#for i in range(17):
#    directory.append('m' + str(i+1))


# 디렉토리별로 돌아가면서 모든 사진들에 대해서 +4장씩 데이터를 증식시킴(클래스별 총 데이터 수가 45*4=180장이상이 됨)
for folder in directory:
    path = 'C:\\Users\\qo989\\anaconda3\\envs\\detection\\darkflow\\hand\\dataset\\'+folder
    print(path)
    file_list = glob.glob(path + '\\*.jpg')
    # imagesPath = [file for file in file_list if file.endswith(".jpg")]
    print(folder + "\n")

    for eachImgPath in file_list:
        img = load_img(eachImgPath)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0

        # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
        for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=path, save_prefix=folder, save_format='jpg'):
            i += 1
            if i > 3:
                break