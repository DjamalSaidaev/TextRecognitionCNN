import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image
import pytesseract as pts
import matplotlib

matplotlib.use("Agg")


class CNN:
    def __init__(self):
        self.model = None
        self.INIT_LR = 0.01
        self.EPOCHS = 100
        self.BS = 32
        self.data = None
        self.H = None
        self.aug = None
        self.inputShape = None
        self.chanDim = None
        self.type = None
        self.eng_labels = [
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
        ]
        self.rus_labels = [
            1040,
            1041,
            1042,
            1043,
            1044,
            1045,
            1025,
            1046,
            1047,
            1048,
            1049,
            1050,
            1051,
            1052,
            1053,
            1054,
            1055,
            1056,
            1057,
            1058,
            1059,
            1060,
            1061,
            1062,
            1063,
            1064,
            1065,
            1066,
            1067,
            1068,
            1069,
            1070,
            1071,
        ]

    def load_data(self, data):
        self.data = data

    def build(self, width, height, depth):
        self.model = Sequential()
        self.inputShape = (height, width, depth)
        self.chanDim = -1
        if K.image_data_format() == "channels_first":
            self.inputShape = (depth, height, width)
            self.chanDim = 1

        # слои CONV => RELU => POOL
        self.model.add(Conv2D(64, (3, 3), padding="same", input_shape=self.inputShape))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Conv2D(64, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # слои (CONV => RELU) * 2 => POOL
        self.model.add(Conv2D(128, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Conv2D(128, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # слои(CONV= > RELU) *3 = > POOL
        self.model.add(Conv2D(256, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Conv2D(256, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Conv2D(256, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # слои(CONV= > RELU) *3 = > POOL
        self.model.add(Conv2D(512, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Conv2D(512, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Conv2D(512, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # слои FC = > RELU
        self.model.add(Flatten())
        self.model.add(Dense(4096))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1000))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization(axis=self.chanDim))
        self.model.add(Dropout(0.5))

        # классификатор softmax
        self.model.add(Dense(len(self.data.lb.classes_)))
        self.model.add(Activation("softmax"))
        self.compile_model()

    def compile_model(self):
        print("[INFO] training network...")
        opt = SGD(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
        self.model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    def evaluating_model(self):
        # оцениваем нейросеть
        print("[INFO] evaluating network...")
        score = self.model.evaluate(
            self.data.testX, self.data.testY, batch_size=32, verbose=1
        )
        print()
        print(u"Оценка теста: {}".format(score[0]))
        print(u"Оценка точности модели: {}".format(score[1]))

    def make_plots(self):
        self.evaluating_model()
        print("СТРОИМ ГРАФИКИ")
        # График точности модели
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        ax1.plot(self.H.history["accuracy"])
        ax1.plot(self.H.history["val_accuracy"])
        ax1.set_title("model accuracy")
        ax1.set_ylabel("accuracy")
        ax1.set_xlabel("epoch")
        ax1.legend(["train", "test"], loc="upper left")
        # График оценки loss
        ax2.plot(self.H.history["loss"])
        ax2.plot(self.H.history["val_loss"])
        ax2.set_title("model loss")
        ax2.set_ylabel("loss")
        ax2.set_xlabel("epoch")
        ax2.legend(["train", "test"], loc="upper left")
        plt.savefig("plot.jpg")
        print("СОХРАНИЛИ ГРАФИКИ!!!")

    def trainCNN(self):
        # обучаем нейросеть
        print("TRAINING IS STARTED!!!")
        # Set a learning rate reduction
        datagen = ImageDataGenerator(
            zoom_range=0.2,
            rotation_range=20,
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,
        )  # randomly shift images vertically (fraction of total height)
        datagen.fit(self.data.trainX)

        learning_rate_reduction = ReduceLROnPlateau(
            monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.00001
        )
        # fit the model on the batches generated by datagen.flow()---most parameters similar to model.fit
        self.H = self.model.fit_generator(
            datagen.flow(self.data.trainX, self.data.trainY, batch_size=self.BS),
            steps_per_epoch=64,
            epochs=self.EPOCHS,
            callbacks=[learning_rate_reduction],
            validation_data=(self.data.testX, self.data.testY),
            verbose=1,
        )
        # сохраняем обученную модель
        self.model.save("rus_letters.h5")
        print("Сохранили обученную модель!")

    def load_fitted_model(self, path):
        self.model = load_model(path)
        if path == "eng_letters.h5":
            self.type = 1
        elif path == "rus_letters.h5":
            self.type = 2

    def predict(self, img):
        otvet = ""
        img_arr = np.expand_dims(img, axis=0)
        img_arr = 1 - img_arr / 255.0
        if self.type == 1:
            img_arr[0] = np.rot90(img_arr[0], 3)
            img_arr[0] = np.fliplr(img_arr[0])
        img_arr = img_arr.reshape((1, 28, 28, 1))
        result = self.model.predict_classes([img_arr])
        if self.type == 1:
            otvet = chr(self.eng_labels[result[0]])
        elif self.type == 2:
            otvet = chr(self.rus_labels[result[0]])
        return otvet

    def letters_extract(self, image_file, out_size=28):
        img = cv.imread(image_file)
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
        # img_erode = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        # contours, hierarchy = cv.findContours(img_erode, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        green_low = np.array([45, 100, 50])
        green_high = np.array([75, 255, 255])
        curr_mask = cv.inRange(hsv_img, green_low, green_high)
        hsv_img[curr_mask > 0] = [75, 255, 200]
        # converting the HSV image to Gray inorder to be able to apply contouring
        rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)
        gray = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)

        ret, threshold = cv.threshold(gray, 90, 255, 0)

        contours, hierarchy = cv.findContours(
            threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        output = img.copy()
        letters = []
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv.boundingRect(contour)
            # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            # hierarchy[i][0]: the index of the next contour of the same level
            # hierarchy[i][1]: the index of the previous contour of the same level
            # hierarchy[i][2]: the index of the first child
            # hierarchy[i][3]: the index of the parent
            if hierarchy[0][idx][3] == 0:
                cv.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                letter_crop = gray[y : y + h, x : x + w]
                # print(letter_crop.shape)

                # Resize letter canvas to square
                size_max = max(w, h)
                letter_square = 255 * np.ones(
                    shape=[size_max, size_max], dtype=np.uint8
                )
                if w > h:
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos : y_pos + h, 0:w] = letter_crop
                elif w < h:
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos : x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                # Resize letter to 28x28 and add letter and its X-coordinate
                letters.append(
                    (
                        x,
                        w,
                        cv.resize(
                            letter_square,
                            (out_size, out_size),
                            interpolation=cv.INTER_AREA,
                        ),
                    )
                )

        # Sort array in place by X-coordinate
        letters.sort(key=lambda t: t[0], reverse=False)
        cv.imwrite("temp_image.jpg", output)
        return letters

    def predict1(self, image_file):
        img = cv.imread(image_file)
        pts.pytesseract.tesseract_cmd = r"E:\Tesseract-OCR\tesseract.exe"
        cv.imwrite("temp.jpg", img)
        text = pts.image_to_string(Image.open(image_file), lang="rus+eng")
        os.remove("temp.jpg")
        return text

    def img_to_str1(self, image_file):
        letters = self.letters_extract(image_file)
        s_out = ""
        for i in range(len(letters)):
            dn = (
                letters[i + 1][0] - letters[i][0] - letters[i][1]
                if i < len(letters) - 1
                else 0
            )
            s_out += self.predict(letters[i][2])
            if dn > 10:
                s_out += " "
        s_out = s_out.replace("0", "O")
        s_out = s_out.replace("1", "I")
        print(s_out)
        return s_out

    def img_to_str(self, image_file):
        letters = self.letters_extract(image_file)
        s_out = self.predict1(image_file)
        s_out = s_out.replace("\n", " ")
        print(s_out)
        return s_out
