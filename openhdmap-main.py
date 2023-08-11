#!/usr/bin/env python3
import os
import cv2
import time
import threading
import contextlib
import depthai as dai
from datetime import datetime

import config
from config import trigger_counter, capture_distance
from pipeline_utils import worker
from gps_utils import gps_setup, gps_distance_capture
from camera_capture_loop_utils import run_camera_loop

# Get the date and time of the mission
config.mission_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") #Update the variable in config.py
os.makedirs('captures/' + config.mission_datetime, exist_ok=True)

device_infos = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')

config.set_barrier(len(device_infos))  # Block continuing until all devices ready

with contextlib.ExitStack() as exit_stack:
    queues_dict = {}
    threads = []

    for device_info in device_infos:
        time.sleep(1) # Currently required due to XLink race issues
        thread = threading.Thread(target=worker, args=(device_info, exit_stack, queues_dict))

        thread.start()
        threads.append(thread)

    counters = {device_info.getMxId(): 0 for device_info in device_infos}
    counters_dict = {}

    for device_info in device_infos:
        counters_dict[device_info.getMxId()] = 0

    config.barrier.wait() # Ensure all devices ready before proceeding

    gps = gps_setup()
    gps_thread = threading.Thread(target=gps_distance_capture, args=(gps, trigger_counter, capture_distance, device_infos, queues_dict))
    gps_thread.start()

    while True:
        run_camera_loop(device_infos, queues_dict, counters_dict)

        key = cv2.waitKey(1)
        if key == ord('q'):
            gps_quit_flag = True  # Signal the GPS thread to stop
            break

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Wait for the GPS thread to finish
    gps_thread.join()

    cv2.destroyAllWindows()

print('Devices closed')