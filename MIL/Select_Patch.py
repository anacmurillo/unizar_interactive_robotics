import numpy
import cv2
import sys
import gzip
from Scene import *
import cPickle
from PyQt4.QtGui import *
from PyQt4.QtCore import *
path = '/media/iglu/Data/Data/'
paths = '/media/iglu/Data/DatasetIglu'
users = ["user"+n.__str__() for n in xrange(1,11)]
actions = ['point_1', 'point_2', 'point_3', 'point_4', 'point_5', 'point_6', 'point_7', 'point_8', 'point_9', 'point_10','show_1',  'show_2',  'show_3',  'show_4',  'show_5',  'show_6',  'show_7',  'show_8',  'show_9',  'show_10']


class MyDialog(QDialog):
    def __init__(self,parent = None):
        super(MyDialog, self).__init__(parent)
        self.list = [path + user + "_" + action + ".gpz" for action in actions for user in users]
        self.counter_FM = 0
        self.counter_P = 0
        files = gzip.open(self.list[self.counter_FM], 'r')
        self.FM = cPickle.load(files)
        files.close()
        self.buttonY = QPushButton('Yes', self)
        self.buttonY.clicked.connect(self.handleYesbutton)
        self.labeledit = QLineEdit()

        self.buttonN = QPushButton('No', self)
        self.buttonN.clicked.connect(self.handleNobutton)

        self.buttonQ = QPushButton('Exit', self)
        self.buttonQ.clicked.connect(self.handleExitbutton)

        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        self.grid.addWidget(self.buttonY, 4, 4)
        self.grid.addWidget(self.buttonN,4,5)
        self.grid.addWidget(self.buttonQ,4,6)
        self.grid.addWidget(self.labeledit, 4, 7)
        self.grid.setRowStretch(2, 1)

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(2)
        self.vbox.addLayout(self.grid)

        self.hbox = QHBoxLayout()
        self.hbox.addLayout(self.vbox)
        self.hbox.addStretch(2)
        self.changed = False
        self.setLayout(self.hbox)
        self.cvImage_ori = self.FM.Candidate_patch[self.counter_P].patch.copy()
        self.cvImage = self.FM.Candidate_patch[self.counter_P].patch.copy()
        self.labeledit.setText(self.FM.Candidate_patch[self.counter_P].Label)
        self.counter_P+=1
        height, width, byteValue = self.cvImage.shape
        byteValue = byteValue * width

        cv2.cvtColor(self.cvImage, cv2.COLOR_BGR2RGB, self.cvImage)

        self.mQImage = QImage(self.cvImage, width, height, byteValue, QImage.Format_RGB888)
        while 'Valid' in self.FM.Candidate_patch[self.counter_P-1].Values.keys():
            self.set_image()
        self.installEventFilter(self)


    def set_FM(self):
        if self.changed:
            files = gzip.open(self.list[self.counter_FM], 'w')
            cPickle.dump(self.FM, files, -1)
            files.close()
        self.changed=False
        self.counter_FM+=1
        print self.list[self.counter_FM]
        files = gzip.open(self.list[self.counter_FM], 'r')
        self.FM = cPickle.load(files)
        files.close()

    def set_image(self):
        if self.counter_P != len(self.FM.Candidate_patch):
            self.image_changed(self.FM.Candidate_patch[self.counter_P].patch.copy(),True)
            self.labeledit.setText(self.FM.Candidate_patch[self.counter_P].Label)
            self.counter_P+=1
        else:
            self.set_FM()
            while(self.FM.Candidate_patch is None or len(self.FM.Candidate_patch) == 0):
                self.set_FM()
            self.counter_P = 0
            self.image_changed(self.FM.Candidate_patch[self.counter_P].patch.copy(), True)
            self.labeledit.setText(self.FM.Candidate_patch[self.counter_P].Label)
            self.counter_P += 1


    def handleYesbutton(self):
        self.FM.Candidate_patch[self.counter_P-1].Values["Valid"]=True
        self.changed = True
        self.set_image()


    def handleNobutton(self):
        self.FM.Candidate_patch[self.counter_P-1].Values["Valid"] = False
        self.changed = True
        self.set_image()

    def handleExitbutton(self):
        exit(0)
        # self.answer = 'Q'

    def getshape(self):
        height, width, byteValue = self.cvImage.shape
        return height,width

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.mQImage)
        painter.end()

    def image_changed(self,img,bool):
        self.cvImage = img

        height, width, byteValue = self.cvImage.shape
        byteValue = byteValue * width
        if bool:
            cv2.cvtColor(self.cvImage, cv2.COLOR_BGR2RGB, self.cvImage)

        self.mQImage = QImage(self.cvImage, width, height, byteValue, QImage.Format_RGB888)
        self.update()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = MyDialog()
    x, y = w.getshape()
    w.resize(300,300)
    w.show()
    app.exec_()