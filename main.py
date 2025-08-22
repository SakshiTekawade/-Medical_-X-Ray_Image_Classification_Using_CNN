# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\opencv\ui\Gujarathi_lang_recognition\demo2.ui'
# Created by: PyQt5 UI code generator 5.11.3
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os, json, math
import numpy as np

# Use TF-Keras (compatible with TF 2.10 on Python 3.10)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Extra libs from original (not required for core flow, but retained)
import cv2, scipy, imutils, csv
import scipy.io as sio
from imutils import contours
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ===================== YOUR DATA FOLDERS (raw strings) =====================
TRAIN_DIR = r"C:\Users\saurabh\Downloads\29_Covid19DetectionusingCNN\29_Covid19DetectionusingCNN\TrainingDataset"
TEST_DIR  = r"C:\Users\saurabh\Downloads\29_Covid19DetectionusingCNN\29_Covid19DetectionusingCNN\TestingDataset"
# ==========================================================================

IMG_SIZE = (128, 128)
BATCH_SIZE = 16                      # a bit larger batch for stability/speed
EPOCHS = 20                          # increase if needed for minority class
MODEL_JSON_PATH = "model.json"
MODEL_WEIGHTS_PATH = "model.h5"
BEST_WEIGHTS_PATH = "best.h5"
CLASS_INDEX_MAP_PATH = "class_indices.json"

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setObjectName("BrowseImage")
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont()
        font.setFamily("Courier New")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setObjectName("Classify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 111, 16))
        self.label.setObjectName("label")
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setObjectName("Training")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)

        self.file = None  # selected image path

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrowseImage.setText(_translate("MainWindow", "Browse Image"))
        self.label_2.setText(_translate("MainWindow", "            COVID-19 DETECTION"))
        self.Classify.setText(_translate("MainWindow", "Classify"))
        self.label.setText(_translate("MainWindow", "Recognized Class"))
        self.Training.setText(_translate("MainWindow", "Training"))

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)"
        )
        if fileName:
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
            self.textEdit.setText(f"Loaded: {os.path.basename(fileName)}")

    def classifyFunction(self):
        if not self.file:
            self.textEdit.setText("Please browse and select an image first.")
            return
        if not (os.path.exists(MODEL_JSON_PATH) and os.path.exists(MODEL_WEIGHTS_PATH)):
            self.textEdit.setText("Model files not found. Please click Training first.")
            return

        with open(MODEL_JSON_PATH, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(MODEL_WEIGHTS_PATH)
        print("Loaded model from disk")

        # Correct label order from saved mapping
        try:
            with open(CLASS_INDEX_MAP_PATH, "r") as f:
                class_indices = json.load(f)  # e.g., {"Covid":0,"Normal":1} OR {"Normal":0,"Covid":1}
            inv = {v: k for k, v in class_indices.items()}
            label_list = [inv[i] for i in range(len(inv))]
        except Exception as e:
            print("class_indices.json missing; falling back to default order", e)
            label_list = ["Covid", "Normal"]

        img = image.load_img(self.file, target_size=IMG_SIZE)
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        probs = loaded_model.predict(x)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        pred_label = label_list[pred_idx] if pred_idx < len(label_list) else f"Class {pred_idx}"
        conf = float(np.max(probs))
        print("Prediction:", probs, "=>", pred_label, conf)
        self.textEdit.setText(f"{pred_label} ({conf:.3f})")

    def trainingFunction(self):
        self.textEdit.setText("Training under process...")

        if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(TEST_DIR):
            self.textEdit.setText("Training/Testing directories not found. Check TRAIN_DIR/TEST_DIR.")
            return

        # === Model (same architecture/flow as original) ===
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # === Data generators (consistent preprocessing + modest augmentation) ===
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
        )
        print("Training class_indices:", training_set.class_indices)

        test_set = test_datagen.flow_from_directory(
            TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
        )
        print("Testing class_indices:", test_set.class_indices)

        # Save mapping for correct inference label order
        with open(CLASS_INDEX_MAP_PATH, "w") as f:
            json.dump(training_set.class_indices, f)

        # === Class weighting (handles imbalance Normal≫Covid) ===
        y = training_set.classes
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print("Class weights:", class_weight)

        # === Callbacks: early stop + best weights ===
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(BEST_WEIGHTS_PATH, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
        ]

        steps_per_epoch = math.ceil(training_set.samples / training_set.batch_size)
        validation_steps = math.ceil(test_set.samples / test_set.batch_size)

        history = model.fit(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=test_set,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

        # Ensure best weights are used
        if os.path.exists(BEST_WEIGHTS_PATH):
            model.load_weights(BEST_WEIGHTS_PATH)
            print("Restored best validation weights from checkpoint.")

        # Save final model and weights
        model_json = model.to_json()
        with open(MODEL_JSON_PATH, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(MODEL_WEIGHTS_PATH)
        print("Saved model to disk (model.json, model.h5, class_indices.json).")

        # === Quick evaluation + detailed metrics on your test set ===
        test_loss, test_acc = model.evaluate(test_set, steps=validation_steps, verbose=1)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        test_set.reset()
        probs = model.predict(test_set, steps=validation_steps, verbose=1)
        y_true = test_set.classes
        y_pred = np.argmax(probs, axis=1)[:len(y_true)]

        idx2label = {v: k for k, v in test_set.class_indices.items()}
        target_names = [idx2label[i] for i in range(len(idx2label))]
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

        self.textEdit.setText("Training complete. Model saved. Check console for metrics.")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
