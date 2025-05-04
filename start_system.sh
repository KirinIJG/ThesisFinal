#!/bin/bash

# Check if main.py is already running
if pgrep -f main.py > /dev/null
then
    echo "Blindspot System is already running. Exiting."
    exit 1
fi

# Navigate to the project directory
cd /home/blindspot/Documents/Thesis/Thesis/main/speed_yolo_dir_back_thread_cuda_yolo_map_integratemapandcamera_centralized_speedplease_pathprojection_newmap_deepsort_log1_relativemotion_optimize_ui_orient_flip/Blindspot/main

# Launch the system
/usr/bin/python3 main.py

