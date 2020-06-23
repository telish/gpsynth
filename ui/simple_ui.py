import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot

from gpsynth.synthesizer import GPSynth, kernel_for_string, all_kernels
from gpsynth.audio_output import RealtimeAudio


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()

        self.cb = QComboBox()
        self.cb.addItems(all_kernels)

        button = QPushButton('Play')
        button.clicked.connect(self.on_click)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(500)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self.slider_changed)

        self.line_edit_left = QLineEdit()
        self.line_edit_left.setFixedWidth(40)
        self.line_edit_left.setText('0.01')
        self.line_edit_left.setValidator(QDoubleValidator())

        self.line_edit_right = QLineEdit()
        self.line_edit_right.setFixedWidth(40)
        self.line_edit_right.setText('2.0')
        self.line_edit_right.setValidator(QDoubleValidator())

        grid.addWidget(self.cb, 0, 2)
        grid.addWidget(QLabel('lengthscale = '), 1, 0)
        grid.addWidget(self.line_edit_left, 1, 1)
        grid.addWidget(self.slider, 1, 2)
        grid.addWidget(self.line_edit_right, 1, 3)
        self.label_lengthscale = QLabel('0.0')
        self.label_lengthscale.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.label_lengthscale, 2, 2)
        grid.addWidget(button, 3, 2)

        self.setLayout(grid)

        self.setWindowTitle("GPSynth Simple GUI")
        self.slider_changed()

        self.audio_output = RealtimeAudio()

    @pyqtSlot()
    def on_click(self):
        left = float(self.line_edit_left.text())
        right = float(self.line_edit_right.text())
        lengthscale = left + (right - left) * self.slider.value() / 1000.0
        kernel = kernel_for_string(self.cb.currentText(), lengthscale)
        gpsynth = GPSynth(kernel, out_rt=self.audio_output, out_wav=None)
        play_jingle(gpsynth)

    def lengthscale_input(self):
        left = float(self.line_edit_left.text())
        right = float(self.line_edit_right.text())
        max_slider = 1000
        return left + self.slider.value() / max_slider * (right - left)

    @pyqtSlot()
    def slider_changed(self):
        self.label_lengthscale.setText(f'{self.lengthscale_input():.2f}')


def play_jingle(gpsynth):
    t = 1. / 8.
    for i in range(16):
        gpsynth.note(60 + i, t)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())
