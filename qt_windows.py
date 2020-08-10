from PyQt5 import QtWidgets, QtCore
from design import Ui_MainWindow
from design_tree import Ui_MainWindow2
from PyQt5.QtGui import QPalette, QBrush, QImage, QPixmap
from PyQt5.QtCore import QSize, pyqtSignal, QObject
from PyQt5.QtWidgets import QFileSystemModel, QMessageBox
from Speaking import Audio
from DataPreparing import MakeData
from NeuralNet import CNN
import os
import cv2 as cv
import numpy as np


class Communicate(QObject):
    closeApp = pyqtSignal()


class TreeWindow(QtWidgets.QMainWindow):
    def __init__(self, main_wind):
        super(TreeWindow, self).__init__()
        self.par = main_wind
        self.ui = Ui_MainWindow2()
        self.ui.setupUi(self)
        self.ui.ok.clicked.connect(self.choose_file)
        self.ui.close.clicked.connect(self.close)
        img_filters = ["*.jpg", "*.JPG"]
        model = QFileSystemModel()
        model.setRootPath(QtCore.QDir.currentPath())
        model.setNameFilters(img_filters)
        model.setNameFilterDisables(0)
        self.ui.treeView.setModel(model)
        self.ui.treeView.setColumnWidth(0, 400)

    def choose_file(self):
        if len(self.ui.treeView.selectedIndexes()) != 0:
            index = self.ui.treeView.selectedIndexes()[0]
            name = self.ui.treeView.model().fileName(index)
            if (
                (name.find(".jpg") != -1)
                or (name.find(".JPG") != -1)
            ):
                self.par.file_path = self.ui.treeView.model().filePath(index)
                # вырабатываем сигнал о закрытии окна
                self.par.signal_from_tree_window.closeApp.emit()
                self.close()
        else:
            pass


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.language = ""
        # создаём переменные для модели
        self.au = Audio()
        self.output = ""
        self.md = None
        self.nn = CNN()

        self.w = 689
        self.h = 488

        # создаём флаги
        self.flag_choose = 0
        self.flag_speak = 0

        # создаем главное окно
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # создаем окно выбора изображения
        self.file_tree = TreeWindow(self)

        # загружаем изображение на задний фон главного окна
        # oImage = QImage("fon1.jpg")
        # sImage = oImage.scaled(QSize(self.w, self.h))
        # palette = QPalette()
        # palette.setBrush(QPalette.Window, QBrush(sImage))
        # self.setPalette(palette)

        self.file_path = ""
        self.ui.but_load.clicked.connect(self.but_load_clicked)

        self.ui.but_run.clicked.connect(self.but_run_clicked)

        self.ui.but_sound.clicked.connect(self.but_sound_clicked)

        self.ui.but_exit.clicked.connect(self.but_exit_clicked)

        self.ui.horizontalLayout.addWidget(self.ui.img_label)
        self.ui.verticalLayout.addWidget(self.ui.img_label)
        # обработчик событий для общения между двумя окнами
        self.signal_from_tree_window = Communicate()
        self.signal_from_tree_window.closeApp.connect(self.choose_signal)

        self.ui.comboBox.activated[str].connect(self.combo_active)
        self.language = self.ui.comboBox.currentText()

    def resizeEvent(self, event):
        width = self.size().width()
        height = self.size().height()

        koefW = float(width) / float(self.w)
        koefH = float(height) / float(self.h)

        self.w = width
        self.h = height

        self.ui.but_load.setGeometry(
            round(self.ui.but_load.frameGeometry().x() * koefW),
            round(self.ui.but_load.frameGeometry().y() * koefH),
            self.ui.but_load.frameGeometry().width(),
            self.ui.but_load.frameGeometry().height(),
        )

        self.ui.but_run.setGeometry(
            round(self.ui.but_run.frameGeometry().x() * koefW),
            round(self.ui.but_run.frameGeometry().y() * koefH),
            self.ui.but_run.frameGeometry().width(),
            self.ui.but_run.frameGeometry().height(),
        )

        self.ui.but_sound.setGeometry(
            round(self.ui.but_sound.frameGeometry().x() * koefW),
            round(self.ui.but_sound.frameGeometry().y() * koefH),
            self.ui.but_sound.frameGeometry().width(),
            self.ui.but_sound.frameGeometry().height(),
        )

        self.ui.comboBox.setGeometry(
            round(self.ui.comboBox.frameGeometry().x() * koefW),
            round(self.ui.comboBox.frameGeometry().y() * koefH),
            self.ui.comboBox.frameGeometry().width(),
            self.ui.comboBox.frameGeometry().height(),
        )

        self.ui.but_exit.setGeometry(
            round(self.ui.but_exit.frameGeometry().x() * koefW),
            round(self.ui.but_exit.frameGeometry().y() * koefH),
            self.ui.but_exit.frameGeometry().width(),
            self.ui.but_exit.frameGeometry().height(),
        )

        self.ui.img_label.setGeometry(
            round(self.ui.img_label.frameGeometry().x() * koefW),
            round(self.ui.img_label.frameGeometry().y() * koefH),
            round(self.ui.img_label.frameGeometry().width() * koefW),
            round(self.ui.img_label.frameGeometry().height() * koefH),
        )

        tmp_img = self.make_image(
            self.ui.img_label.frameGeometry().width(),
            self.ui.img_label.frameGeometry().height(),
        )
        if len(tmp_img) != 0:
            cv.imwrite("qwerty.jpg", tmp_img)
            pixmap = QPixmap("qwerty.jpg")
            self.ui.img_label.setPixmap(pixmap)
            os.remove("qwerty.jpg")

        # oImage = QImage("fon1.jpg")
        # sImage = oImage.scaled(QSize(self.w, self.h))
        # palette = QPalette()
        # palette.setBrush(QPalette.Window, QBrush(sImage))
        # self.setPalette(palette)

    def combo_active(self, text):
        if text == "Выберите язык":
            QMessageBox.question(
                self, "Сообщение", "Вы не выбрали язык", QMessageBox.Ok, QMessageBox.Ok,
            )
        else:
            self.language = text

    def but_exit_clicked(self):
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        self.close()

    def but_sound_clicked(self):
        if self.flag_speak:
            self.language = self.ui.comboBox.currentText()
            self.au.speech(self.output, self.language)
        else:
            QMessageBox.question(
                self,
                "Сообщение",
                "Не было распознано изображение!",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )

    def make_image(self, width, height):
        output = np.array([])
        if os.path.exists("temp_image.jpg"):
            img = cv.imread("temp_image.jpg")
            scale_percent = 1
            # width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dsize = (width, height)
            output = cv.resize(img, dsize, interpolation=cv.INTER_AREA)
        return output

    def but_run_clicked(self):
        if self.flag_choose:
            if self.language == "ENG":
                self.nn.load_fitted_model("eng_letters.h5")
            elif self.language == "RUS":
                self.nn.load_fitted_model("rus_letters.h5")
            # выбор Tesseract (img_to_str) или своя модель (img_to_str1)
            self.output = self.nn.img_to_str(self.file_path)

            tmp_img = self.make_image(
                self.ui.img_label.frameGeometry().width(),
                self.ui.img_label.frameGeometry().height(),
            )
            if len(tmp_img) != 0:
                cv.imwrite("qwerty.jpg", tmp_img)
                pixmap = QPixmap("qwerty.jpg")
                self.ui.img_label.setPixmap(pixmap)
                os.remove("qwerty.jpg")

            self.ui.img_label.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
            )
            self.ui.img_label.resize(self.ui.img_label.sizeHint())
            self.flag_speak = 1
            self.flag_choose = 0

            QMessageBox.question(
                self,
                "Распозанный текст",
                f"{self.output}",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            self.file_path = ""
        else:
            QMessageBox.question(
                self,
                "Сообщение",
                "Не было выбрано изображение",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )

    def but_load_clicked(self):
        self.flag_speak = 0
        self.output = ""
        self.file_tree.show()

    def choose_signal(self):
        self.flag_choose = 1
        QMessageBox.question(
            self, "Сообщение", "Изображение выбрано", QMessageBox.Ok, QMessageBox.Ok,
        )


# nn = CNN()
# md = MakeData("eng_dataset")
# md.work_with_data(1)
# nn.load_data(md)
# nn.build(width=28, height=28, depth=1)
# nn.trainCNN()
# nn.make_plots()
