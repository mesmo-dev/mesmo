import sys

from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QMessageBox, QPushButton, QLabel, QLineEdit)


class CreateNewUser(QWidget):

    def __init__(self):
        super().__init__()
        self.initializeUI()  # Call our function used to set up window

    def initializeUI(self):
        """
        Initialize the window and display its contents to the screen
        """
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle('3.2 - Create New User')
        self.displayWidgetsToCollectInfo()
        self.show()

    def displayWidgetsToCollectInfo(self):
        # Create label for image
        new_user_image = "imag/new_user_icon.png"

        try:
            with open(new_user_image):
                new_user = QLabel(self)
                pixmap = QPixmap(new_user_image)
                pixmap = pixmap.scaled(128, 128)
                new_user.setPixmap(pixmap)
                new_user.move(150, 60)
        except FileNotFoundError:
            print("Image not found.")

        login_label = QLabel(self)
        login_label.setText("create new account")
        login_label.move(110, 20)
        login_label.setFont(QFont('Arial', 12))

        # Username and fullname labels and line edit widgets
        name_label = QLabel("username:", self)
        name_label.move(50, 180)
        self.name_entry = QLineEdit(self)
        self.name_entry.move(130, 180)
        self.name_entry.resize(200, 20)
        name_label = QLabel("full name:", self)
        name_label.move(50, 210)
        name_entry = QLineEdit(self)
        name_entry.move(130, 210)
        name_entry.resize(200, 20)
        # Create password and confirm password labels and line edit widgets
        pswd_label = QLabel("password:", self)
        pswd_label.move(50, 240)
        self.pswd_entry = QLineEdit(self)
        self.pswd_entry.setEchoMode(QLineEdit.Password)
        self.pswd_entry.move(130, 240)
        self.pswd_entry.resize(200, 20)
        confirm_label = QLabel("confirm:", self)
        confirm_label.move(50, 270)
        self.confirm_entry = QLineEdit(self)
        self.confirm_entry.setEchoMode(QLineEdit.Password)
        self.confirm_entry.move(130, 270)
        self.confirm_entry.resize(200, 20)
        # Create sign up button
        sign_up_button = QPushButton("sign up", self)
        sign_up_button.move(100, 310)
        sign_up_button.resize(200, 40)
        sign_up_button.clicked.connect(self.confirmSignUp)

    def confirmSignUp(self):
        pswd_text = self.pswd_entry.text()
        confirm_text = self.confirm_entry.text()
        if pswd_text != confirm_text:
            # Display messagebox if passwords don't match
            QMessageBox.warning(self, "Error Message", " The passwords you entered do not match. Please try again.",
                                QMessageBox.Close,
                                QMessageBox.Close)
        else:
            # If passwords match, save passwords to file and return to login
            # and test if you can log in with new user information.
            with open("../files/users.txt", 'a+') as f:
                f.write(self.name_entry.text() + " ")
                f.write(pswd_text + "\n")

            self.close()


# Run program
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CreateNewUser()
    sys.exit(app.exec_())
