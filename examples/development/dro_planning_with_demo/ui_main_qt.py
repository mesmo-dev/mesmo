import sys
import time
import mesmo
import os
import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px

from module_result_visualization.results_visualization import create_result_webpage
from module_optimal_battery_sizing_placement.data_interface import data_battery_sizing_placement
from module_optimal_battery_sizing_placement.deterministic_acopf_planning import \
    deterministic_acopf_battery_placement_sizing

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtCore import *
from module_ui.Registration import CreateNewUser
from PyQt5.QtWebEngineWidgets import *



class Welcome_page(QWidget):
    switch_window = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        # Initialize the window and display its contents to the screen.
        self.setGeometry(400, 300, 1260, 580)
        self.setWindowTitle('MESMO')
        self.displayButton()
        self.displayLabels()
        # self.show()

    def displayLabels(self):
        text = QLabel(self)
        text.setText("MESMO Optimal Planning Toolkit: ")
        text.move(25, 18)
        image = "imag/mesmo_logo.png"
        try:
            with open(image):
                world_image = QLabel(self)
                pixmap = QPixmap(image)
                world_image.setPixmap(pixmap)
                world_image.move(25, 60)
        except FileNotFoundError:
            print("Image not found.")

    def displayButton(self):
        button = QPushButton('Start DEMO', self)
        button.clicked.connect(self.buttonClicked)
        button.move(300, 10)  # arrange button

    def buttonClicked(self):
        # print("The window has been closed.")
        # self.close()
        self.switch_window.emit()


class Login(QWidget):  # Inherits QWidget

    switch_window = pyqtSignal()

    def __init__(self):  # Constructor
        super().__init__()  # Initializer which calls constructor for QWidget
        self.initializeUI()  # Call function used to set up window

    def initializeUI(self):
        self.setGeometry(790, 400, 400, 300)
        self.setWindowTitle('Login page')
        self.displayWidgets()
        # self.show()

    def displayWidgets(self):
        # Create name label and line edit widgets
        # QLabel("Please enter the password", self).move(100, 10)

        name_label = QLabel("username:", self)
        name_label.move(30, 60)
        self.name_entry = QLineEdit(self)
        self.name_entry.move(110, 60)
        self.name_entry.resize(220, 20)
        password_label = QLabel("password:", self)
        password_label.move(30, 90)
        self.password_entry = QLineEdit(self)
        self.password_entry.move(110, 90)
        self.password_entry.resize(220, 20)

        sign_in_button = QPushButton('login', self)
        sign_in_button.move(100, 140)
        sign_in_button.resize(230, 40)
        sign_in_button.clicked.connect(self.clickLogin)

        # Display show password checkbox
        show_pswd_cb = QCheckBox("show password", self)
        show_pswd_cb.move(110, 115)
        show_pswd_cb.stateChanged.connect(self.showPassword)
        show_pswd_cb.toggle()
        show_pswd_cb.setChecked(False)

        # Display sign up label and push button
        not_a_member = QLabel("not a member?", self)
        not_a_member.move(30, 250)
        sign_up = QPushButton("sign up", self)
        sign_up.move(160, 245)
        sign_up.clicked.connect(self.createNewUser)

    def clickLogin(self):
        users = {}  # Create empty dictionary to store user information
        # Check if users.txt exists, otherwise create new file
        try:
            with open("files/users.txt", 'r') as f:
                for line in f:
                    user_fields = line.split(" ")
                    username = user_fields[0]
                    password = user_fields[1].strip('\n')
                    users[username] = password
        except FileNotFoundError:
            print("The file does not exist. Creating a new file.")
            f = open("files/users.txt", "w")

        username = self.name_entry.text()
        password = self.password_entry.text()

        if (username, password) in users.items():
            QMessageBox.information(self, "Login Successful!", "Login Successful!",
                                    QMessageBox.Ok, QMessageBox.Ok)
            self.switch_window.emit()
            # self.close()  # close program
        else:
            QMessageBox.warning(self, "Error Message", "The username or password is incorrect.",
                                QMessageBox.Close, QMessageBox.Close)

    def showPassword(self, state):
        # If checkbox is enabled, view password. Else, mask password so others cannot see it.
        if state == Qt.Checked:
            self.password_entry.setEchoMode(QLineEdit.Normal)
        else:
            self.password_entry.setEchoMode(QLineEdit.Password)

    def createNewUser(self):
        self.create_new_user_dialog = CreateNewUser()
        self.create_new_user_dialog.show()

    # def closeEvent(self, event):
    #   # Display a QMessageBox when asking the user if they want to quit the program. - it doesn't work yet
    #   # Set up message box
    #   quit_msg = QMessageBox.question(self, "Quit Application?", "Are you sure you want to Quit?",
    #                                   QMessageBox.No | QMessageBox.Yes, QMessageBox.Yes)
    #
    #   if quit_msg == QMessageBox.Yes:
    #    event.accept()  # accept the event and close the application
    #   else:
    #    event.ignore()  # ignore the close event

    def clearEntries(self):
        sender = self.sender()
        if sender.text() == 'Clear':
            self.password_entry.clear()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()  # create default constructor for QWidget
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle('MESMO Optimal Distribution Network Planning Toolkit')
        self.showMaximized()
        self.createMenu()
        self.createToolBar()
        self.createDockWidget()
        self.show()

    def createMenu(self):
        # Create actions for file menu
        exit_act = QAction('Exit', self)
        exit_act.setShortcut('Ctrl+Q')
        exit_act.triggered.connect(self.close)

        new_project_act = QAction('New demo project', self)
        new_project_act.setShortcut('Ctrl+P')
        new_project_act.triggered.connect(self.selectDemoProject)

        # Create menubar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        # Create file menu and add actions
        file_menu = menu_bar.addMenu('Project')
        file_menu.addAction(new_project_act)
        file_menu.addAction(exit_act)

    def selectDemoProject(self):
        self.demo_project_selection = DemoSelectionMenu()
        self.demo_project_selection.show()

    def createToolBar(self):
        tool_bar = QToolBar("Main Toolbar")
        tool_bar.setIconSize(QSize(16, 16))
        self.addToolBar(tool_bar)

        # Add actions to toolbar
        button_action = QAction(QIcon("bug.png"), "Your button", self)
        button_action.setStatusTip("This is your button")
        button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)

        tool_bar.addAction(button_action)

        self.setStatusBar(QStatusBar(self))

    def onMyToolBarButtonClick(self, s):
     print("click", s)

    def createDockWidget(self):
        dock_widget = QDockWidget()
        dock_widget.setWindowTitle("Operation Dock")
        dock_widget.setAllowedAreas(Qt.AllDockWidgetAreas)
        # Set main widget for the dock widget
        dock_widget.setWidget(QTextEdit())
        # Set initial location of dock widget in main window
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)


class DemoSelectionMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.initializeUI()

    def initializeUI(self):
        self.setGeometry(700, 300, 400, 230)
        self.setWindowTitle('demo project selection')
        self.displayWidgets()
        self.show()

    def displayWidgets(self):
        # Create label and button widgets
        title = QLabel("Demo Project Selection")
        title.setFont(QFont('Arial', 17))
        question = QLabel("Which DEMO functionality you would like to choose?")
        # Create horizontal layouts
        title_h_box = QHBoxLayout()
        title_h_box.addStretch()
        title_h_box.addWidget(title)
        title_h_box.addStretch()

        ratings = ["ESS placement", "empty option", "empty option"]
        # Create checkboxes and add them to horizontal layout, and add stretchable
        # space on both sides of the widgets
        ratings_h_box = QHBoxLayout()
        ratings_h_box.setSpacing(60)  # Set spacing between in widgets in horizontal layout
        ratings_h_box.addStretch()

        for rating in ratings:
            rate_label = QLabel(rating, self)
            ratings_h_box.addWidget(rate_label)

        ratings_h_box.addStretch()
        cb_h_box = QHBoxLayout()
        cb_h_box.setSpacing(100)  # Set spacing between in widgets in horizontal layout
        # Create button group to contain checkboxes
        scale_bg = QButtonGroup(self)
        cb_h_box.addStretch()

        for cb in range(len(ratings)):
            scale_cb = QCheckBox(str(cb), self)
            cb_h_box.addWidget(scale_cb)
            scale_bg.addButton(scale_cb)

        cb_h_box.addStretch()
        # Check for signal when checkbox is clicked
        scale_bg.buttonClicked.connect(self.checkboxClicked)

        close_button = QPushButton("Start Demo", self)
        close_button.setFixedHeight(60)
        close_button.clicked.connect(self.DisplayResults)
        # Create vertical layout and add widgets and h_box layouts
        v_box = QVBoxLayout()

        v_box.addLayout(title_h_box)
        v_box.addWidget(question)
        v_box.addStretch(1)
        v_box.addLayout(ratings_h_box)
        v_box.addLayout(cb_h_box)
        v_box.addStretch(2)

        # Set main layout of selection window
        self.setLayout(v_box)

        combobox1_title = QLabel("Test case:")
        combobox1 = QComboBox()
        combobox1.addItem('IEEE 4-bus test case')
        combobox1.addItem('IEEE 123-bus test case')
        combobox1.addItem('Singapore Tanjong Pagar case')
        combobox1.addItem('Singapore 6-bus test case')

        combobox2_title = QLabel("Output mode:")
        combobox1.activated.connect(self.activated)
        combobox1.currentTextChanged.connect(self.text_changed)
        combobox1.currentIndexChanged.connect(self.index_changed)

        combobox2 = QComboBox()
        combobox2.addItems(['minimal', 'tall', 'grande'])

        space = QLabel(" ")
        v_box.addWidget(combobox1_title)
        v_box.addWidget(combobox1)
        v_box.addWidget(combobox2_title)
        v_box.addWidget(combobox2)
        v_box.addWidget(space)
        v_box.addWidget(close_button)

    def activated(Self, index):
        print("Activated index:", index)

    def text_changed(self, s):
        print("Text changed:", s)

    def index_changed(self, index):
        print("Index changed", index)

    def checkboxClicked(self, cb):
        print("{} Selected.".format(cb.text()))

    def DisplayResults(self):
        self.display_results = DisplayResults()
        self.display_results.show()


class DisplayResults(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        vbox = QVBoxLayout(self)

        #self.progress_bar = ProgressBar()

        # Settings.
        scenario_name = 'paper_2021_zhang_dro'
        mesmo.data_interface.recreate_database()

        # Obtain data.
        data_set = data_battery_sizing_placement(
            os.path.join(os.path.dirname(os.path.normpath(__file__)),
                         'module_optimal_battery_sizing_placement',
                         'test_case_customized')
        )

        # Get results path.
        results_path = mesmo.utils.get_results_path(__file__, scenario_name)

        # Get standard form of stage 1.
        optimal_sizing_problem = deterministic_acopf_battery_placement_sizing(scenario_name, data_set)

        optimal_sizing_problem.optimization_problem.solve()
        results = optimal_sizing_problem.optimization_problem.get_results()

        #result_web = create_result_webpage()
        #result_web.app.run_server(debug=True)

        results_html = pd.DataFrame.from_dict(results.values()).to_html()

        with open(os.path.join(os.path.dirname(os.path.normpath(__file__)),
                         'files','optimization_result',
                         'res.html'), mode='w') as file_object:
            print(results_html, file=file_object)

        self.webEngineView = QWebEngineView()
        #self.webEngineView.setUrl(QUrl("http://127.0.0.1:8050/"))
        self.loadPage()


        vbox.addWidget(self.webEngineView)

        self.setLayout(vbox)
        self.setGeometry(100, 200, 1700, 700)
        self.setWindowTitle('Results Visualization')
        self.show()

    def loadPage(self):

        #self.setUrl(QUrl("http://127.0.0.1:8050/"))
        
        with open('files/optimization_result/res.html', 'r') as f:

            html = f.read()
            self.webEngineView.setHtml(html)

class ProgressBar(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # creating progress bar
        self.pbar = QProgressBar(self)

        # setting its geometr  y
        self.pbar.setGeometry(30, 40, 200, 25)

        # creating push button
        self.btn = QPushButton('Start', self)

        # changing its position
        self.btn.move(40, 80)

        # adding action to push button
        self.btn.clicked.connect(self.doAction)

        # setting window geometry
        self.setGeometry(300, 300, 280, 170)

        # setting window action
        self.setWindowTitle("Python")

        # showing all the widgets
        self.show()

        # when button is pressed this method is being called

    def doAction(self):
        # setting for loop to set value of progress bar
        for i in range(101):
            # slowing down the loop
            time.sleep(0.02)

            # setting value to progress bar
            self.pbar.setValue(i)


class Controller:
    def __init__(self):
        pass

    def show_welcome_page(self):
        self.welcome_page = Welcome_page()
        self.welcome_page.switch_window.connect(self.show_login)
        self.welcome_page.show()

    def show_login(self):
        self.login = Login()
        self.login.switch_window.connect(self.show_main_window)
        self.welcome_page.close()
        self.login.show()

    def show_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.login.close()


# Run program
if __name__ == '__main__':
    app = QApplication(sys.argv)

    controller = Controller()
    controller.show_welcome_page()

    # test individual page
    # mainWindow = ProgressBar()
    sys.exit(app.exec_())
