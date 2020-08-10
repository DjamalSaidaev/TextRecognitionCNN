import idx2numpy
from imutils import paths
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import cv2 as cv


class MakeData:
    def __init__(self, path):
        self.testX = []
        self.testY = []
        self.trainX = []
        self.trainY = []
        self.data = []  # данные
        self.labels = []  # метки
        self.lb = None
        self.path_to_data = path

    def open_eng_data(self):
        self.trainX = idx2numpy.convert_from_file(
            "eng_dataset/emnist-byclass-train-images-idx3-ubyte"
        )
        self.trainY = idx2numpy.convert_from_file(
            "eng_dataset/emnist-byclass-train-labels-idx1-ubyte"
        )

        self.testX = idx2numpy.convert_from_file(
            "eng_dataset/emnist-byclass-test-images-idx3-ubyte"
        )
        self.testY = idx2numpy.convert_from_file(
            "eng_dataset/emnist-byclass-test-labels-idx1-ubyte"
        )

        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 28, 28, 1))
        self.testX = np.reshape(self.testX, (self.testX.shape[0], 28, 28, 1))

        k = 5
        self.trainX = self.trainX[: self.trainX.shape[0] // k]
        self.trainY = self.trainY[: self.trainY.shape[0] // k]
        self.testX = self.testX[: self.testX.shape[0] // k]
        self.testY = self.testY[: self.testY.shape[0] // k]

        # Normalize
        self.trainX = self.trainX.astype(np.float32)
        self.trainX /= 255.0
        self.testX = self.testX.astype(np.float32)
        self.testX /= 255.0
        # конвертируем метки из целых чисел в векторы
        self.lb = LabelBinarizer()
        self.trainY = self.lb.fit_transform(self.trainY)
        self.testY = self.lb.transform(self.testY)

    def open_rus_data(self):
        # берём пути к изображениям и рандомно перемешиваем
        imagePaths = sorted(list(paths.list_images(self.path_to_data)))
        random.seed(42)
        random.shuffle(imagePaths)
        # цикл по изображениям
        for i, imagePath in enumerate(imagePaths):
            # загружаем изображение, меняем размер на 28x28 пикселей (без учёта соотношения сторон)
            # добавляем в список
            # переводим изображение в черно-белое
            image = cv.imread(imagePath)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
            img_erode = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
            img_erode = cv.resize(img_erode, (28, 28), interpolation=cv.INTER_AREA)
            self.data.append(img_erode)

            # извлекаем метку класса из пути к изображению и обновляем
            # список меток
            label = imagePath.split(os.path.sep)[-2]
            self.labels.append(int(label))
        # масштабируем интенсивности пикселей в диапазон[0, 1]

        self.data = np.array(self.data, dtype="float")
        self.data = self.data.reshape(self.data.shape[0], 28, 28, 1)
        k = 10
        self.data = self.data[: self.data.shape[0] // k]
        self.data /= 255.0
        self.labels = np.array(self.labels, dtype="int")
        self.labels = self.labels[: self.labels.shape[0] // k]

        # разбиваем данные на обучающую и тестовую выборки, используя 75% данных
        # для обучения и оставшиеся 25% для тестирования
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(
            self.data, self.labels, test_size=0.25, random_state=42
        )
        self.lb = LabelBinarizer()
        self.trainY = self.lb.fit_transform(self.trainY)
        self.testY = self.lb.transform(self.testY)

    # инициализируем данные и метки
    def work_with_data(self, temp):
        if temp == 1:
            self.open_eng_data()
        elif temp == 2:
            self.open_rus_data()
