import tensorflow as tf  # install

##########################################################
##########################LOAD DATA#######################
TRAIN_DATA_ZIP_PATH = "Train_Data.zip" 

import zipfile
with zipfile.ZipFile(TRAIN_DATA_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall("vietnamese_hcr/raw") # giai nen data

import pathlib
current_directory_path = pathlib.Path("./vietnamese_hcr").absolute() # chuyen duong dan thanh tuyet doi
#TEST
print(current_directory_path)

import os

#tao duong dan toi cac file moi de sau nay luu du lieu
DATA_PATH = os.path.join(str(current_directory_path), "data")
TEST_FOLDER = os.path.join(DATA_PATH, "test")
TRAIN_FOLDER = os.path.join(DATA_PATH, "train")

RAW_FOLDER = os.path.join(str(current_directory_path), "raw")
# sau nay vo du an thuc te thi la file .txt
TRAIN_JSON = os.path.join(RAW_FOLDER, "labels.json")


#tao cac file dua tren cac duong dan tren
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(TEST_FOLDER):
    os.makedirs(TEST_FOLDER)
if not os.path.exists(TRAIN_FOLDER):
    os.makedirs(TRAIN_FOLDER)

#kiem tra file json luu du lieu doi chieu
import json
with open(TRAIN_JSON, 'r', encoding='utf8') as f:
    train_labels = json.load(f) # cac nhan cua train data

char_list = set()
for label in train_labels.values():
    char_list.update(set(label))
char_list = sorted(char_list) # lay ra danh sach ky tu trong cac nhan roi sap xep

#luu char list
char_list_tmp = "".join(char_list)
print(char_list_tmp)
with open("./char_list.txt", "w", encoding="utf-8") as file:
    file.write(char_list_tmp)

#TEST
print("char list length: " ,len(char_list))
print("".join(char_list))

#ma hoa tung tu cua dau ra thanh index
def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print("No found in char_list :", char)
    
    return dig_lst

#TEST
encode_to_labels("Dit con me thang Tuan Lieu fff")

############XU LY DUONG DAN ANH VA NHAN##########

train_image_path = []

for item in pathlib.Path(RAW_FOLDER).glob('**/*'):
    if item.is_file() and item.suffix not in [".json"]:
        train_image_path.append(str(item))

#TEST
print(train_image_path[:5])

#tao ra dic luu thong tin duong dan anh va nhan cua anh do

dict_filepath_label = {} # luu cac doi tuong chua thong tin la duong dan anh va nhan cua chung
raw_data_path = pathlib.Path(os.path.join(RAW_FOLDER))
for item in raw_data_path.glob('**/*.*'):
    file_name = str(os.path.basename(item))
    if file_name != "labels.json":
        label = train_labels[file_name] # tim kiem gia tri cua nhan dua theo duong dan anh
        dict_filepath_label[str(item)] = label

#TEST
#print(dict_filepath_label)

#tim nhan dai nhat
label_lens = []
for label in dict_filepath_label.values():
    label_lens.append(len(label))
max_label_len = max(label_lens)

#TEST
print(max_label_len)

############XU LY ANH#############

all_image_paths = list(dict_filepath_label.keys()) # lay ra danh sach duong dan toi anh

import cv2 # install

#lay thong tin kich thuoc anh
widths = []
heigths = []
for image_path in all_image_paths:
    img = cv2.imread(image_path)
    (heigth, width, _) = img.shape
    heigths.append(heigth)
    widths.append(width)

min_height = min(heigths)
max_heigth = max(heigths)
min_width = min(widths)
max_width = max(widths)

#TEST
print((min_height, max_heigth, min_width, max_width))

#####CHUAN BI DU LIEU##############
#chia file data thanh file train va file xac thuc

from sklearn.model_selection import train_test_split #install

test_size = 0.2 # ty le file train va file xac thuc
train_image_paths, val_image_paths = train_test_split(all_image_paths, test_size=test_size, random_state=42)


####CHUAN BI DU LIEU DE TRAIN#############

TIME_STEPS = 240

import numpy as np

training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []  #gia tri cac nhan dung
resize_max_width = 0
i=0

for train_img_path in train_image_paths:
    #doc va convert sang dinh dang gray
    img = cv2.cvtColor(cv2.imread(train_img_path), cv2.COLOR_BGR2GRAY)

    height, width = img.shape

    #resize lai anh
    img = cv2.resize(img, (int(118/height*width), 118))

    height, width = img.shape

    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]

    img = np.pad(img, ((0,0),(0, 2167 - width)), 'median') #chuan hoa hinh anh voi cac hinh anh co kich thuoc khac nhau

    #lam mo hinh anh de giam nhieu
    img = cv2.GaussianBlur(img, (5,5), 0)

    #xu ly voi cac hinh anh sang ko deu 
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    #them chieu cho hinh anh(toi uu voi he thong)
    img = np.expand_dims(img, axis=2)

    #chuan hoa tung pixel anh
    img = img/255.

    label = dict_filepath_label[train_img_path]

    orig_txt.append(label)
    train_label_length.append(len(label))

    train_input_length.append(TIME_STEPS)
    training_img.append(img)

    #convert cac ky tu sang ma so index trong charlist
    training_txt.append(encode_to_labels(label))
    i+=1
    if(i%500 ==  0):
        print("Has processed trained {} files".format(i))

#TEST
print(resize_max_width)

###############################################
# import matplotlib.pyplot as plt
# for i in range(1):
#     plt.figure(figsize=(15,2))
#     plt.imshow(training_img[i][:,:,0], cmap="gray")
#     plt.show()

###############################################

###########XU LY ANH TEST##################

valid_img = [] #danh sach anh de xac thuc
valid_txt = []
valid_input_length = []
valid_label_length = [] #
valid_orig_txt = [] 
i=0

for val_img_path in val_image_paths:

    img = cv2.cvtColor(cv2.imread(val_img_path), cv2.COLOR_BGR2GRAY)

    #
    img = cv2.resize(img, (int(118/height*width),118))

    if img.shape[1] > resize_max_width:
        resize_max_width = img.shape[1]

    #
    img = np.pad(img, ((0,0), (0,2167-width)), 'median')

    img = cv2.GaussianBlur(img, (5,5), 0)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    img = np.expand_dims(img, axis=2)

    img = img / 255.

    ##################chua bi nhan cua du lieu xac thuc de so sanh

    label = dict_filepath_label[val_img_path]

    valid_orig_txt.append(label)
    valid_label_length.append(len(label))

    valid_txt.append(encode_to_labels(label))
    i+=1
    if (i%500 == 0):
        print("Has processed test {} file".format(i))
     
    #############
    valid_input_length.append(TIME_STEPS)
    valid_img.append(img)

    #TEST
    #print(height, width)

#TEST

####################################
# import matplotlib.pyplot as plt
# for i in range(1):
#     plt.figure(figsize=(15,2))
#     plt.imshow(valid_img[i][:,:,0], cmap="gray")
#     plt.show()
######################################

max_label_len = TIME_STEPS

######################################
#dem de tat ca cac nhan co cung do dai de train nhat quan 
from keras.utils import pad_sequences

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value=0)
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value=0)

#TEST 
print(train_padded_txt[0])

##############################################################
#######################MODEL BUILD############################

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

#TEST
print(act_model.summary())


#define hinh dang nhan dau vao cho ctc

labels = Input(name = 'the_labels', shape=[max_label_len], dtype='float32')

input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# ctc func nhan vao doi so roi tra ve ctc_bach_cost
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
model = Model(inputs=[inputs, labels, input_length, label_length], outputs = loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

#
callbacks = [
    TensorBoard(
        log_dir='./logs', # thư mục nơi nhật ký TensorBoard sẽ được lưu trữ.
        histogram_freq=10, #Tần suất (tính bằng kỷ nguyên) để tính toán biểu đồ của trọng số và kích hoạt.
        profile_batch=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch"), #Tần suất (theo "kỷ nguyên" hoặc "đợt") để ghi nhật ký vào TensorBoard.
    ModelCheckpoint(  #lưu trọng số của mô hình trong quá trình đào tạo, tùy chọn dựa trên tổn thất xác thực.
        filepath=os.path.join('checkpoint_weights.hdf5'), # Đường dẫn đến nơi lưu trọng số điểm kiểm tra.
        monitor='val_loss', #Số liệu để theo dõi. Điểm kiểm tra mô hình sẽ được lưu khi số liệu này được cải thiện.
        save_best_only=True,
        save_weights_only=True,
        verbose=1),
    EarlyStopping( #ngừng đào tạo khi số liệu được theo dõi đã ngừng cải thiện
        monitor='val_loss', 
        min_delta=1e-8,
        patience=20, #Số lượng kỷ nguyên không cải thiện sau đó việc đào tạo sẽ bị dừng.
        restore_best_weights=True,  
        verbose=1),
    ReduceLROnPlateau( #giảm tốc độ học khi số liệu được giám sát ngừng cải thiện
        monitor='val_loss', 
        min_delta=1e-8,
        factor=0.2,
        patience=10, #Số lượng kỷ nguyên không được cải thiện, sau đó tốc độ học sẽ giảm.
        verbose=1)
]

callbacks_list = callbacks

#TEST
print(model.summary())


###########################################
#chuan bi du lieu de train

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

#chuan bi du lieu xac thuc
valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)


##########TRAINING#################

batch_size = 32
epochs = 100

history = model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], 
          y=np.zeros(len(training_img)),
          batch_size=batch_size, 
          epochs = epochs,
          validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]),
          verbose = 1, callbacks = callbacks_list)

#################################################################
#######################TEST DATA#############################

# lay ra file luu trong so cua cach mang than kinh tot nhat
act_model.load_weights(os.path.join('checkpoint_weights.hdf5'))

NO_PREDICTS = 1
OFFSET = 0

prediction = act_model.predict(valid_img) #chay du lieu test

#lay ra gia tri du doan cua fife test o dang index sau do decode
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                               greedy=True)[0][0])

all_predictions = []
i = 0
for x in out:
    print("Original_text = ", valid_orig_txt[i+OFFSET])
    print("Predicted text = ", end = '')
    pred = ""
    for p in x:
        if int(p) != -1:
            pred += char_list[int(p)]
    
    print(pred)
    all_predictions.append(pred)
    i+=1

##########END#############
print("Con chim to vailon")
