import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QRadioButton, QCheckBox, QFileDialog, QProgressBar, QButtonGroup, QGridLayout, QMessageBox, QToolTip, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QSize, Qt
import os
from utils.loader import load_raw_accel_file
from utils.feature_extraction import extract_features_with_start_times
from utils.classifier import load_xgboost_classifier, predict, get_label_encoder
from utils.summarizer import summarize_predictions, merge_summary_files
from utils.nonwear_cleaner import process_nonwear_times
import pandas as pd
from os import path
import multiprocessing

def get_relative_path(relpath):
    return path.abspath(path.join(path.dirname(__file__), relpath))


class ToddlerAccelApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QGridLayout()

        # Documentation Link
        doc_label = QLabel('For full documentation and user instructions, please click <a href="https://github.com/LettsE/ToddlerMachineLearning">here</a>')
        doc_label.setOpenExternalLinks(True)
        doc_label.setAlignment(Qt.AlignCenter)

        # Input Folder
        input_info = QLabel("ℹ️")
        input_info.setToolTip("Click 'Browse' to select the folder where your input files are located.")
        input_info.mousePressEvent = lambda event: QToolTip.showText(event.globalPos(), input_info.toolTip())
        input_label = QLabel("Input folder:")
        self.input_line_edit = QLineEdit()
        input_button = QPushButton("Browse folders")
        input_button.clicked.connect(self.browseInputFolder)

        # Output Folder
        output_info = QLabel("ℹ️")
        output_info.setToolTip("Click 'Browse' to select the folder where your output files will be saved.")
        output_info.mousePressEvent = lambda event: QToolTip.showText(event.globalPos(), output_info.toolTip())
        output_label = QLabel("Output folder:")
        self.output_line_edit = QLineEdit()
        output_button = QPushButton("Browse folders")
        output_button.clicked.connect(self.browseOutputFolder)

        # Outcome assessment options
        outcome_label = QLabel("Which outcomes do you want to assess?")
        self.total_physical_activity_rb = QRadioButton("Non-volitional Movement, Sedentary Time, Total Physical Activity")
        self.light_moderate_activity_rb = QRadioButton("Non-volitional Movement, Sedentary Time, Light Physical Activity, Moderate-To-Vigorous Physical Activity")
       
        outcome_group = QButtonGroup(self)
        outcome_group.addButton(self.total_physical_activity_rb)
        outcome_group.addButton(self.light_moderate_activity_rb)
        

        # Non-wear removal method
        non_wear_label = QLabel("Which method of nonwear removal would you like to use?")
        self.logbook_diary_rb = QRadioButton("Logbook/Diary")
        self.none_rb = QRadioButton("None")

        non_wear_group = QButtonGroup(self)
        non_wear_group.addButton(self.logbook_diary_rb)
        non_wear_group.addButton(self.none_rb)

        self.logbook_diary_rb.toggled.connect(self.toggleLogbookFileVisibility)

        # Logbook file selection
        logbook_info = QLabel("ℹ️")
        logbook_info.setToolTip("Click 'Browse' to select the logbook file if using the 'Logbook/Diary' method. The file needs to be a .csv file with headers studyid,WearTimeStart,WearTimeEnd.")
        logbook_info.mousePressEvent = lambda event: QToolTip.showText(event.globalPos(), logbook_info.toolTip())
        logbook_label = QLabel("Logbook file:")
        self.logbook_line_edit = QLineEdit()
        logbook_button = QPushButton("Browse files")
        logbook_button.clicked.connect(self.browseLogbookFile)

        # Hide by default
        logbook_label.setVisible(False)
        self.logbook_line_edit.setVisible(False)
        logbook_button.setVisible(False)
        logbook_info.setVisible(False)

        self.logbook_widgets = (logbook_label, self.logbook_line_edit, logbook_button, logbook_info)

        # Run models button
        run_models_button = QPushButton("Run models")
        run_models_button.clicked.connect(self.runModels)

        # Status label
        self.status_label = QLabel()
        self.status_label.setVisible(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Running GIF
        self.running_gif_label = QLabel()
        movie = QMovie(get_relative_path("assets/RunningGif08March2025.gif"))  
        movie.setScaledSize(QSize(225, 155))
        self.running_gif_label.setMovie(movie)
        self.running_gif_label.setVisible(False)
        self.running_gif_label.setAlignment(Qt.AlignCenter)

        # Footer
        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignCenter)
        copyright_label = QLabel('Copyright © 2025 McMaster University, Hamilton, Ontario, Canada. Little Movers Activity Analysis, authored by Elyse Letts and Joyce Obeid, is the copyright of McMaster University. By using Little Movers Activity Analysis you agree to the <a href="https://github.com/LettsE/ToddlerMachineLearning/blob/main/LICENSE">Terms of Use</a>.')
        copyright_label.setOpenExternalLinks(True)
        copyright_label.setWordWrap(True) 
        copyright_label.setFixedWidth(650)
        footer_layout.addWidget(copyright_label)
        
        
        # Main layout setup
        main_layout.addWidget(doc_label, 0, 0, 1, 4)
        main_layout.addWidget(input_info, 1, 0)
        main_layout.addWidget(input_label, 1, 1)
        main_layout.addWidget(self.input_line_edit, 1, 2)
        main_layout.addWidget(input_button, 1, 3)

        main_layout.addWidget(output_info, 2, 0)
        main_layout.addWidget(output_label, 2, 1)
        main_layout.addWidget(self.output_line_edit, 2, 2)
        main_layout.addWidget(output_button, 2, 3)

        main_layout.addWidget(outcome_label, 3, 0, 1, 4)
        main_layout.addWidget(self.total_physical_activity_rb, 4, 0, 1, 4)
        main_layout.addWidget(self.light_moderate_activity_rb, 5, 0, 1, 4)
  
        main_layout.addWidget(non_wear_label, 11, 0, 1, 4)
        main_layout.addWidget(self.logbook_diary_rb, 12, 0, 1, 4)
        main_layout.addWidget(self.none_rb, 16, 0, 1, 4)

        main_layout.addWidget(logbook_info, 17, 0)
        main_layout.addWidget(logbook_label, 17, 1)
        main_layout.addWidget(self.logbook_line_edit, 17, 2)
        main_layout.addWidget(logbook_button, 17, 3)

        main_layout.addWidget(self.running_gif_label, 18, 0, 1, 4)
        main_layout.addWidget(self.status_label, 19, 0, 1, 4)
        main_layout.addWidget(self.progress_bar, 20, 0, 1, 4)
        main_layout.addWidget(run_models_button, 21, 3)

        # Set layout
        container_layout = QVBoxLayout()
        container_layout.addLayout(main_layout)
        container_layout.addLayout(footer_layout)

        self.setLayout(container_layout)
        self.setWindowTitle('Little Movers Activity Analysis ©')
        self.show()

    def browseInputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_line_edit.setText(folder)

    def browseOutputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_line_edit.setText(folder)

    def browseLogbookFile(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Logbook File", "", "All Files (*)")
        if file:
            self.logbook_line_edit.setText(file)

    def toggleLogbookFileVisibility(self, checked):
        for widget in self.logbook_widgets:
            widget.setVisible(checked)

    def getSelectedData(self):
        # Collect selections made by the user
        selected_data = {
            "input_folder": self.input_line_edit.text(),
            "output_folder": self.output_line_edit.text(),
            "outcome": None,
            "result_outputs": [],
            "non_wear_method": None,
            "logbook_file": self.logbook_line_edit.text() if self.logbook_diary_rb.isChecked() else None
        }

        if self.total_physical_activity_rb.isChecked():
            selected_data["outcome"] = "NVM_TPA_SED"
        elif self.light_moderate_activity_rb.isChecked():
            selected_data["outcome"] = "NVM_LPA_MVPA_SED"

        if self.logbook_diary_rb.isChecked():
            selected_data["non_wear_method"] = "Logbook"
        elif self.none_rb.isChecked():
            selected_data["non_wear_method"] = "None"

        return selected_data

    def runModels(self):
        if not self.areAllSelectionsMade():
            QMessageBox.warning(self, "Missing Selection", "Missing one or more selections")
            return
        
        # Start the GIF and show the progress bar
        self.running_gif_label.setVisible(True)
        self.status_label.setVisible(True)
        self.running_gif_label.movie().start()
        self.progress_bar.setVisible(True)

        multiprocessing.freeze_support()

        self.thread = QtCore.QThread(self)
        self.worker = Worker(self.getSelectedData())
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.onPipelineFinished)
        self.worker.progressChanged.connect(self.onPipelineProgressed)
        self.thread.started.connect(self.worker.run)
        self.thread.start()


    def onPipelineProgressed(self, count, status):
        print("Progress update", count)
        self.progress_bar.setValue(count)
        self.status_label.setText(status)

    def onPipelineFinished(self):
        self.running_gif_label.setVisible(False)
        self.progress_bar.setVisible(False)

    def areAllSelectionsMade(self):
        if (self.input_line_edit.text() and self.output_line_edit.text() and
            (self.total_physical_activity_rb.isChecked() or self.light_moderate_activity_rb.isChecked()) and
            (self.logbook_diary_rb.isChecked() or self.none_rb.isChecked())):
            return True
        return False
    

class Worker(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal()

    def __init__(self, selectedData):
        super().__init__()
        self._stopped = True
        self.selectedData = selectedData

    def run(self):
        self.progressChanged.emit(0, "Loading the model...")
        
        if self.selectedData["outcome"] == "NVM_TPA_SED" :
            model_path = get_relative_path("models/NVM-0_SED-1_TPA-2_5sIntPos.json")
            label_classes = ['NVM', 'SED', 'TPA']
        elif self.selectedData["outcome"] == "NVM_LPA_MVPA_SED" :
            model_path = get_relative_path("models/NVM-0_SED-1_LPA-2_MVPA-3_5sIntPos.json")
            label_classes = ['NVM', 'SED', 'LPA', 'MVPA']
        else:
            self.progressChanged.emit(100, "Error: No model selected!")
            return  # Stop execution if no selection is made


        model = load_xgboost_classifier(model_path)
        label_encoder = get_label_encoder(label_classes)
        self.progressChanged.emit(100, "Model loaded successfully!")

        gt3x_folder = self.selectedData['input_folder']
        files = os.listdir(gt3x_folder)
        nonwear_method = self.selectedData['non_wear_method']
        logbook_file = self.selectedData['logbook_file']
        output_folder = self.selectedData['output_folder']



        summary_results = pd.DataFrame()

        for idx, filename in enumerate(files):
             file_path = gt3x_folder + "/" + filename
             if os.path.isfile(file_path) and filename.endswith(".gt3x"):
                self.progressChanged.emit(round((idx)/len(files)*100), "Removing nonwear for " + filename + "...")
                data = load_raw_accel_file(file_path)
                
                studyid = os.path.splitext(filename)[0]
                
                daily_summary, trimmed_data = process_nonwear_times(data, nonwear_method, logbook_file, output_folder, studyid)
                
                self.progressChanged.emit(round((idx)/len(files)*100), "Nonwear removed for  " + filename)

                self.progressChanged.emit(round((idx)/len(files)*100), "Extracting features for " + filename + "...")
                features, start_times = extract_features_with_start_times(trimmed_data, epoch=5, hz=30)

                self.progressChanged.emit(round((idx)/len(files)*100), "Features extracted for " + filename)

                predictions = predict(model, features, start_times, label_encoder)
                predictions.to_csv(os.path.join(self.selectedData['output_folder'], filename + "_predictions.csv"))

                self.progressChanged.emit(round((idx+1.0)/len(files)*100), "File " + filename + " successfully predicted!")

                summary = summarize_predictions(predictions, epoch=5)
                summary['studyid'] = studyid
                summary_results = pd.concat([summary_results, summary], axis=0)

                # Save the summary results to CSV after each file
                summary_results.to_csv(os.path.join(self.selectedData['output_folder'], "by_day_by_participants.csv"), mode='a', header=not os.path.exists(os.path.join(self.selectedData['output_folder'], "by_day_by_participants.csv")))
                summary_results = pd.DataFrame()  
                merge_summary_files(self.selectedData['output_folder'])

        self.progressChanged.emit(100, "Complete!")
        self._stopped = True
        self.finished.emit()

    def stop(self):
        self._stopped = True

def main():
    app = QApplication(sys.argv)
    ex = ToddlerAccelApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()
