# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(689, 488)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setStyleSheet("")
        self.centralwidget.setObjectName("centralwidget")
        self.but_run = QtWidgets.QPushButton(self.centralwidget)
        self.but_run.setGeometry(QtCore.QRect(10, 260, 200, 41))
        self.but_run.setMaximumSize(QtCore.QSize(302, 202))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.but_run.setFont(font)
        self.but_run.setStyleSheet("background-color: rgb(186,182,184);\n"
"border-style: solid;\n"
"border-width:1px;\n"
"border-radius:10px;\n"
"border-color: white;\n"
"max-width:300px;\n"
"max-height:200px;\n"
"min-width:10px;\n"
"min-height:10px;")
        self.but_run.setObjectName("but_run")
        self.but_load = QtWidgets.QPushButton(self.centralwidget)
        self.but_load.setGeometry(QtCore.QRect(10, 130, 200, 41))
        self.but_load.setMaximumSize(QtCore.QSize(302, 202))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.but_load.setFont(font)
        self.but_load.setStyleSheet("background-color: rgb(186,182,184);\n"
"border-style: solid;\n"
"border-width:1px;\n"
"border-radius:10px;\n"
"border-color: white;\n"
"max-width:300px;\n"
"max-height:200px;\n"
"min-width:10px;\n"
"min-height:10px;")
        self.but_load.setObjectName("but_load")
        self.but_sound = QtWidgets.QPushButton(self.centralwidget)
        self.but_sound.setGeometry(QtCore.QRect(10, 390, 200, 41))
        self.but_sound.setMaximumSize(QtCore.QSize(302, 202))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.but_sound.setFont(font)
        self.but_sound.setStyleSheet("background-color: rgb(186,182,184);\n"
"border-style: solid;\n"
"border-width:1px;\n"
"border-radius:10px;\n"
"border-color: white;\n"
"max-width:300px;\n"
"max-height:200px;\n"
"min-width:10px;\n"
"min-height:10px;")
        self.but_sound.setObjectName("but_sound")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(10, 10, 200, 41))
        self.comboBox.setMaximumSize(QtCore.QSize(302, 202))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.comboBox.setFont(font)
        self.comboBox.setStyleSheet("background-color: rgb(186,182,184);\n"
"border-style: solid;\n"
"border-width:1px;\n"
"border-radius:10px;\n"
"border-color: white;\n"
"max-width:300px;\n"
"max-height:200px;\n"
"min-width:10px;\n"
"min-height:10px;")
        self.comboBox.setEditable(False)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.but_exit = QtWidgets.QPushButton(self.centralwidget)
        self.but_exit.setGeometry(QtCore.QRect(480, 390, 200, 41))
        self.but_exit.setMaximumSize(QtCore.QSize(302, 202))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.but_exit.setFont(font)
        self.but_exit.setStyleSheet("background-color:rgb(186,182,184);\n"
"border-style: solid;\n"
"border-width:1px;\n"
"border-radius:10px;\n"
"border-color: white;\n"
"max-width:300px;\n"
"max-height:200px;\n"
"min-width:10px;\n"
"min-height:10px;")
        self.but_exit.setObjectName("but_exit")
        self.img_label = QtWidgets.QLabel(self.centralwidget)
        self.img_label.setGeometry(QtCore.QRect(240, 20, 431, 341))
        self.img_label.setText("")
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label.setObjectName("img_label")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(230, 10, 451, 361))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(230, 10, 451, 361))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 689, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.comboBox, self.but_run)
        MainWindow.setTabOrder(self.but_run, self.but_load)
        MainWindow.setTabOrder(self.but_load, self.but_sound)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NScanner"))
        self.but_run.setText(_translate("MainWindow", "Распознать"))
        self.but_load.setText(_translate("MainWindow", "Выбор изображения"))
        self.but_sound.setText(_translate("MainWindow", "Воспроизвести"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Выберите язык"))
        self.comboBox.setItemText(1, _translate("MainWindow", "ENG"))
        self.comboBox.setItemText(2, _translate("MainWindow", "RUS"))
        self.but_exit.setText(_translate("MainWindow", "Выход"))
