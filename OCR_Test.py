import zipfile
import pathlib
import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model
import os
# from OCRSystem import act_model
#from OCRSystem import char_list

TEST_DATA_PATH = "./vietnamese_hcr/data/test"

char_list = set()
char_list_tmp = ""
with open("./char_list.txt", 'r', encoding='utf8') as f:
    char_list_tmp = f.read() # cac nhan cua train data
for c in char_list_tmp:
    char_list.update(set(c))
char_list = sorted(char_list)
#TEST
print(char_list_tmp)
print(len(char_list))


test_image_paths = []
for item in pathlib.Path(TEST_DATA_PATH).glob('**/*'):
    if item.is_file() and item.suffix not in [".json"]:
        test_image_paths.append(str(item))

test_img = []
i=0
for test_img_path in test_image_paths:

    img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    img = cv2.resize(img,(int(118/height*width),118))
    height, width = img.shape
    img = np.pad(img, ((0,0),(0, 2167-width)), 'median')
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img = np.expand_dims(img, axis=2)
    img = img/255
    test_img.append(img)

#TEST
# import matplotlib.pyplot as plt
# for i in range(5):
#     plt.figure(figsize=(15,2))
#     plt.imshow(test_img[i][:,:,0], cmap="gray")
#     plt.show()

#chay
test_img = np.array(test_img)
#dung lai model
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
#from keras.utils import to_categorical
#from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#MODEL CRNN, LSTM

inputs = Input(shape=(118, 2167, 1))

#Block 1
x = Conv2D(64, (3,3), padding='same')(inputs)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_1 = x

#Block 2
x = Conv2D(128, (3,3), padding='same')(x)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_2 = x

# Block 3
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_3 = x

# Block4
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_3])
x = Activation('relu')(x)
x_4 = x

# Block5
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_5 = x

# Block6
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_5])
x = Activation('relu')(x)

# Block7
x = Conv2D(1024, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3, 1))(x)
x = Activation('relu')(x)

x = MaxPool2D(pool_size=(3,1))(x)

squeezed = Lambda(lambda x: K.squeeze(x,1))(x)
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)
act_model = Model(inputs, outputs)

#####################
act_model.load_weights('checkpoint_weights.hdf5')
prediction = act_model.predict(test_img)
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                               greedy=True)[0][0])

all_predictions = []
i=0
for x in out:
    print(str(test_image_paths[i]) , " = ", end = '')
    pred = ""
    for p in x:
        if int(p) != -1:
            pred += char_list[int(p)]
    print(pred)
    all_predictions.append(pred)
    i+=1

with open("test_gt.txt", "w", encoding='utf8') as f:
    for image_path, label in zip(test_image_paths, all_predictions):
        image_filename = os.path.basename(image_path)
        f.write(f"{image_filename}\t{label}\n")