import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QListWidget, QPushButton, QVBoxLayout, QWidget, QMainWindow, QMessageBox

class MainWindow(QMainWindow):
    def __init__(self, window_width=600, window_height=500):
        super(MainWindow, self).__init__()

        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2


        self.setWindowTitle("Menu")
        self.setGeometry(x, y, window_width, window_height)

        self.button_1 = QPushButton("Choose Video")
        self.button_2 = QPushButton("Choose Camera")

        self.button_1.setFixedSize(700, 150)
        self.button_2.setFixedSize(700, 150)

        self.button_1.clicked.connect(self.show_videos)
        self.button_2.clicked.connect(self.show_cameras)

        layout = QVBoxLayout()
        layout.addWidget(self.button_1)
        layout.addWidget(self.button_2)

        mainWidget = QWidget()
        mainWidget.setLayout(layout)

        self.setCentralWidget(mainWidget)

    def check_videos(self):
        main_dir = os.getcwd()
        folder_name = "videos"

        f_path = os.path.join(main_dir, folder_name)
        if os.path.isdir(f_path):
            videos = os.listdir(f_path)
            return videos
        else:
            raise FileNotFoundError("Directory 'videos' does not exist.")


    def check_cameras(self, max_cameras=10):
        available_cameras = []

        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()

        return available_cameras

    def choose_items(self, lst):
        listWidget = QListWidget()
        listWidget.addItems(lst)

        button = QPushButton("Show Selected Item")
        button.clicked.connect(lambda: self.on_button_click(listWidget))

        layout = QVBoxLayout()
        layout.addWidget(listWidget)
        layout.addWidget(button)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def show_videos(self):
        global videos, cameras
        cameras = None
        videos = self.check_videos()
        self.setCentralWidget(self.choose_items(videos))

    def show_cameras(self):
        global videos, cameras
        videos = None
        cameras = self.check_cameras()
        self.setCentralWidget(self.choose_items(cameras))

    def on_button_click(self, listWidget):
        global item
        selected_items = listWidget.selectedItems()
        if selected_items:
            item = selected_items[0].text()
            if videos:
                if item in videos:
                    item = f"videos/{item}"
            if cameras:
                if item in cameras:
                    item = int(item)


            self.close()
            # QApplication.quit()
        else:
            self.show_warning()

    def show_warning(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("No selection error")
        msg.setText("You must select an item before proceeding.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()







def main_menu():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
    return item



if __name__ == "__main__":
    main_menu()


    # print(item)



def close_all():
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle("Test window")
    window.setGeometry(300, 250, 300, 300)

    btn = QPushButton("Press", window)
    btn.move(70, 150)
    btn.setFixedWidth(200)

    # Подключение кнопки к функции закрытия приложения
    btn.clicked.connect(app.quit)

    window.show()

    sys.exit(app.exec_())
