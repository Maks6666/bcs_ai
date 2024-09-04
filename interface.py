from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QMessageBox

atgm_ammount = 0
cl_shells_amount = 0
u_shells_ammount = 0
fpv_drones_ammount = 0
manpads_ammount = 0
sam_ammount = 0

def ground_weapons(window_width=800, window_height=600):
    global atgm_ammount, cl_shells_amount, u_shells_ammount, fpv_drones_ammount, manpads_ammount, sam_ammount

    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle('Ground Weapons Group')

    screen = QApplication.primaryScreen().geometry()
    screen_width = screen.width()
    screen_height = screen.height()

    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    window.setGeometry(x, y, window_width, window_height)

    # style for buttons / input fields
    label_font = QFont('Arial', 25, QFont.Bold)
    # input_font = QFont(12)

    # create buttons / input fields
    label1 = QLabel('ATGM amount:', window)
    label1.setFont(label_font)
    text_input1 = QLineEdit(window)


    label2 = QLabel('Cluster shells amount:', window)
    label2.setFont(label_font)
    text_input2 = QLineEdit(window)

    label3 = QLabel('FPV drones amount:', window)
    label3.setFont(label_font)
    text_input3 = QLineEdit(window)

    label4 = QLabel('Unitar shells amount:', window)
    label4.setFont(label_font)
    text_input4 = QLineEdit(window)

    label5 = QLabel('MANPADS ammount:', window)
    label5.setFont(label_font)
    text_input5 = QLineEdit(window)

    label6 = QLabel('SAM ammount:', window)
    label6.setFont(label_font)
    text_input6 = QLineEdit(window)



    submit_button = QPushButton('Submit', window)


    def on_submit():
        global atgm_ammount, cl_shells_amount, u_shells_ammount, fpv_drones_ammount, manpads_ammount, sam_ammount

        atgm_ammount = int(text_input1.text())
        cl_shells_amount = int(text_input2.text())
        u_shells_ammount = int(text_input3.text())
        fpv_drones_ammount = int(text_input4.text())
        manpads_ammount = int(text_input5.text())
        sam_ammount = int(text_input6.text())

        window.close()

    submit_button.clicked.connect(on_submit)

    # Создание макета и добавление виджетов
    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(text_input1)

    layout.addWidget(label2)
    layout.addWidget(text_input2)

    layout.addWidget(label3)
    layout.addWidget(text_input3)

    layout.addWidget(label4)
    layout.addWidget(text_input4)

    layout.addWidget(label5)
    layout.addWidget(text_input5)

    layout.addWidget(label6)
    layout.addWidget(text_input6)

    layout.addWidget(submit_button)

    window.setLayout(layout)
    window.show()
    app.exec_()

    return atgm_ammount, cl_shells_amount, u_shells_ammount, fpv_drones_ammount, manpads_ammount, sam_ammount


def ground_weapons_encoder(atgm_ammount, cl_shells_amount, u_shells_ammount, frv_drones_ammount, manpads_ammount, sam_ammount):
    global atgm, cl_shells, u_shells, fpv_drones, fpv_drones, manpads, sam
    atgm = None
    cl_shells = None
    u_shells = None
    fpv_drones = None
    manpads = None
    sam = None


    if atgm_ammount == 0 or atgm_ammount is None:
        atgm = 0
    elif 0 < atgm_ammount <= 20:
        atgm = 1
    elif 20 < atgm_ammount <= 50:
        atgm = 2
    elif atgm_ammount > 50:
        atgm = 3


    if cl_shells_amount == 0 or cl_shells_amount is None:
        cl_shells = 0
    elif 0 < cl_shells_amount <= 20:
        cl_shells = 1
    elif 20 < cl_shells_amount <= 50:
        cl_shells = 2
    elif cl_shells_amount > 50:
        cl_shells = 3


    if u_shells_ammount == 0 or u_shells_ammount is None:
        u_shells = 0
    elif 0 < u_shells_ammount <= 20:
        u_shells = 1
    elif 20 < u_shells_ammount <= 50:
        u_shells = 2
    elif u_shells_ammount > 50:
        u_shells = 3


    if frv_drones_ammount == 0 or frv_drones_ammount is None:
        fpv_drones = 0
    elif 0 < frv_drones_ammount <= 20:
        fpv_drones = 1
    elif 20 < frv_drones_ammount <= 50:
        fpv_drones = 2
    elif frv_drones_ammount > 50:
        fpv_drones = 3


    if manpads_ammount == 0 or manpads_ammount is None:
        manpads = 0
    elif 0 < manpads_ammount <= 20:
        manpads = 1
    elif 20 < manpads_ammount <= 50:
        manpads = 2
    elif manpads_ammount > 50:
        manpads = 3


    if sam_ammount == 0 or sam_ammount is None:
        sam = 0
    elif 0 < sam_ammount <= 20:
        sam = 1
    elif 20 < sam_ammount <= 50:
        sam = 2
    elif sam_ammount > 50:
        sam = 3

    return atgm, cl_shells, u_shells, fpv_drones, manpads, sam

def single_encoder(item):
    if item == 0 or item is None:
        return 0
    elif 0 < item <= 20:
        return 1
    elif 20 < item <= 50:
        return 2
    elif item > 50:
        return 3



from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QFont
import sys



def vehicles_screen(app, window_width=100, window_height=80, apc_ammount=0, tank_ammount=0, ifv_ammount=0,
                    total_vehicles=0, descision=None):
    window = QWidget()
    window.setWindowTitle('Assault Vehicles')

    screen = QApplication.primaryScreen().geometry()
    screen_width = screen.width()
    screen_height = screen.height()

    x = (screen_width - window_width) - (screen_width // 2)
    y = ((screen_height - window_height) // 2) - 140

    window.setGeometry(x, y, window_width, window_height)



    label_font = QFont('Arial', 25, QFont.Bold)

    label1 = QLabel(f'APC: {apc_ammount}', window)
    label1.setFont(label_font)

    label2 = QLabel(f'Tanks: {tank_ammount}', window)
    label2.setFont(label_font)

    label3 = QLabel(f'IFV: {ifv_ammount}', window)
    label3.setFont(label_font)

    label4 = QLabel(f'Total vehicles: {total_vehicles}', window)
    label4.setFont(label_font)

    label5 = QLabel(f'Decision: {descision}', window)
    label5.setFont(label_font)

    label5_1 = QLabel()
    label5_1.setFont(label_font)

    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(label2)
    layout.addWidget(label3)
    layout.addWidget(label4)
    layout.addWidget(label5)
    layout.addWidget(label5_1)

    window.setLayout(layout)
    window.show()

    return window, label1, label2, label3, label4, label5, label5_1


def troops_screen(app, window_width=100, window_height=80, total_troops=0, descision=None):
    window = QWidget()
    window.setWindowTitle('Troops')

    screen = QApplication.primaryScreen().geometry()
    screen_width = screen.width()
    screen_height = screen.height()

    x = (screen_width - window_width) - (screen_width // 2) + 80
    y = ((screen_height - window_height) // 2) + 150

    window.setGeometry(x, y, window_width, window_height)

    label_font = QFont('Arial', 25, QFont.Bold)



    label1 = QLabel(f'Total troops: {total_troops}', window)
    label1.setFont(label_font)

    label2 = QLabel(f'Decision: {descision}', window)
    label2.setFont(label_font)

    label2_1 = QLabel()
    label2_1.setFont(label_font)



    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(label2)
    layout.addWidget(label2_1)

    window.setLayout(layout)
    window.show()

    return window, label1, label2, label2_1


def specials_screen(app, window_width=100, window_height=80, ev_ammount=0, aa_ammount=0, total_specials_ammount=0,
                    descision=None):
    window = QWidget()
    window.setWindowTitle('Special Vehicles')

    screen = QApplication.primaryScreen().geometry()
    screen_width = screen.width()
    screen_height = screen.height()

    x = (screen_width - window_width) - (screen_width // 2)
    y = (screen_height - window_height) * 0

    window.setGeometry(x, y, window_width, window_height)

    label_font = QFont('Arial', 25, QFont.Bold)

    label1 = QLabel(f'Engineer vehicles: {ev_ammount}', window)
    label1.setFont(label_font)

    label2 = QLabel(f'AA systems: {aa_ammount}', window)
    label2.setFont(label_font)

    label3 = QLabel(f'Total special vehicles: {total_specials_ammount}', window)
    label3.setFont(label_font)

    label4 = QLabel(f'Decision: {descision}', window)
    label4.setFont(label_font)

    label5 = QLabel()
    label5.setFont(label_font)

    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(label2)
    layout.addWidget(label3)
    layout.addWidget(label4)
    layout.addWidget(label5)

    window.setLayout(layout)
    window.show()

    return window, label1, label2, label3, label4, label5


def aviation_screen(app, window_width=100, window_height=80, ah_ammount=0, th_ammount=0, aap_ammount=0, ta_ammount=0,
                    total_flying_units=0, descision=None):
    window = QWidget()
    window.setWindowTitle('Aviation')

    screen = QApplication.primaryScreen().geometry()
    screen_width = screen.width()
    screen_height = screen.height()

    x = 0
    y = screen_height // 2

    window.setGeometry(x, y, window_width, window_height)

    label_font = QFont('Arial', 25, QFont.Bold)

    label1 = QLabel(f'Attack helicopters: {ah_ammount}', window)
    label1.setFont(label_font)

    label2 = QLabel(f'Transport helicopters: {th_ammount}', window)
    label2.setFont(label_font)

    label3 = QLabel(f'Attack planes: {aap_ammount}', window)
    label3.setFont(label_font)

    label4 = QLabel(f'Transport planes: {ta_ammount}', window)
    label4.setFont(label_font)

    label5 = QLabel(f'Total aviation: {total_flying_units}', window)
    label5.setFont(label_font)

    label6 = QLabel(f'Decision: {descision}', window)
    label6.setFont(label_font)

    label7 = QLabel()
    label7.setFont(label_font)

    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(label2)
    layout.addWidget(label3)
    layout.addWidget(label4)
    layout.addWidget(label5)
    layout.addWidget(label6)
    layout.addWidget(label7)

    window.setLayout(layout)
    window.show()

    return window, label1, label2, label3, label4, label5, label6, label7


def artillery_screen(app, window_width=100, window_height=80, tart_ammount=0, spart_ammount=0, total_artillery_units=0,
                     descision=None):
    window = QWidget()
    window.setWindowTitle('Artillery')

    screen = QApplication.primaryScreen().geometry()
    screen_width = screen.width()
    screen_height = screen.height()

    x = ((screen_width - window_width) // 2) + window_width // 2
    y = screen_height

    window.setGeometry(x, y, window_width, window_height)

    label_font = QFont('Arial', 25, QFont.Bold)

    label1 = QLabel(f'TART amount: {tart_ammount}', window)
    label1.setFont(label_font)

    label2 = QLabel(f'SPART amount: {spart_ammount}', window)
    label2.setFont(label_font)

    label3 = QLabel(f'Total artillery: {total_artillery_units}', window)
    label3.setFont(label_font)

    label4 = QLabel(f'Decision: {descision}', window)
    label4.setFont(label_font)

    label5 = QLabel()
    label5.setFont(label_font)

    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(label2)
    layout.addWidget(label3)
    layout.addWidget(label4)
    layout.addWidget(label5)

    window.setLayout(layout)
    window.show()

    return window, label1, label2, label3, label4, label5




