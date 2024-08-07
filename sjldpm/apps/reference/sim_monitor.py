import sys
import os
from pathlib import Path

# Pyside2 
from PySide2.QtCore import Qt, QUrl, Property, Signal, QObject,Slot
from PySide2.QtGui import QGuiApplication, QImage, QPixmap
from PySide2.QtQml import QQmlApplicationEngine
from PySide2.QtWidgets import QApplication
import sys
from pathlib import Path
import os,sys,shutil
from sim_main import load_proc
import rdpm

if __name__ == "__main__":
    
    # === 1. init QML engine
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Expose the instance to QML
    context = engine.rootContext()

    # Load the QML file
    qml = Path(Path(__file__).parent,"sim_monitor.qml").as_posix()
    
    
    # === 2. Do things
    from rdpm.apps.ddpm.sim_main import *
    cfp = Path(__file__).parent
    cfn = Path(__file__).stem
    p_save = Path(cfp,f"{cfn}_save")
    p_save.mkdir(parents=True,exist_ok=True)
    
    pages_model = {}
    
    root_data = Path(Path(rdpm.__file__).parent.parent,"Datasets/CelebAHQ/data256x256")
    
    _,image_path_list = load_proc(root_data,N = 4, only_paths=True)
    
    pages_model["page0"]={
            "info":"dataset images",
            "image_path_list":image_path_list,
            "grid_n_row":2,
            "grid_n_col":2,
        }
    
    
    # === 3. expose to QML engine
    engine.rootContext().setContextProperty("pages_model",pages_model)
    
    # === 4. run app
    engine.load(QUrl.fromLocalFile(qml))
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec_())