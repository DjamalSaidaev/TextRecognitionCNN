import sys
from PyQt5 import QtWidgets
from qt_windows import MyWindow


def main():
    app = QtWidgets.QApplication([])
    application = MyWindow()
    application.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
