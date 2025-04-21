from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files('stumpy', include_py_files=True) # Needed so that it finds gpu_stump.py
datas += collect_data_files('xgboost') # Needed for VERSION
binaries = collect_dynamic_libs("xgboost")