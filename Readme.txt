This code is from our paper "Geological feature-Conscious Large scale factor Super-resolution of well logs using Multi-stage Knowledge transfer"
The Main steps to run the code are:
1. run the "MainFun.m" function in the "preprocessing" folder to complete preprocessing and obtain high-resolution reference data
2- set the data paths and training parameters in the config.py file
3- put the low-resolution and high-resolution CSV files of wells into the correct subfolders under BASE_DIR
4- run the main.py file to start the cascaded super-resolution training and inference
5- during each stage, the output prediction will be saved to the path defined in CASCADE_STAGES
6- after training, use the saved prediction CSVs and metrics from the output paths to evaluate results
7- if needed, modify the parameters (like BATCH_SIZE, EPOCHS, HF_ENHANCEMENT) in config.py and rerun main.py
8- set visualize=True in main.py to automatically generate plots comparing prediction and ground truth