#!/usr/bin/env python3
import os
import csv
import cv2
import time
import serial
import argparse
import threading
import contextlib
import numpy as np
import depthai as dai
from adafruit_gps import GPS
from datetime import datetime
from geopy.distance import distance as geopy_distance

# Neural Net Parameters
nn_path = 'ai_models/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob' # path to AI model
num_of_classes = 21 # define the number of classes in the dataset
nn_shape = 256
nn_jpeg_dims = (2016,1520) # Needs to be rewritten to account for scaling up/down

## Start Function Blocks
def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(nn_shape,nn_shape)
    output = np.array(output) * (255/num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    output_colors[output == 0] = [0,0,0]
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.4,0)

def create_pipeline(mx_id):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(nn_path)
    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A) # Might not be necessary...
    # IMX214 and IMX378-based OAK modules are 12MP sensors:
    # This includes OAK-1, OAK-D, and OAK-D Lite cameras
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    # Comment out below to save full 12MP images at >2.8MB per image...
    cam.setIspScale(1,2) # Halve image resolution using onboard processor
    cam.setFps(10) # Max framerate of NN process anyway
    cam.setPreviewSize(nn_shape,nn_shape)
    cam.setInterleaved(False)
    cam.setPreviewKeepAspectRatio(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    cam.preview.link(detection_nn.input)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName(f"preview-{mx_id}")
    detection_nn.passthrough.link(xoutRgb.input)

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName(f"control-{mx_id}")
    xin.out.link(cam.inputControl)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    cam.still.link(videoEnc.input)

    xoutStill = pipeline.create(dai.node.XLinkOut)
    xoutStill.setStreamName(f"still-{mx_id}")
    videoEnc.bitstream.link(xoutStill.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName(f"nn-{mx_id}")
    detection_nn.out.link(xout_nn.input)

    return pipeline

# Trigger the image capture process
def trigger_photos(device_info, queues_dict):
    mx_id = device_info.getMxId()
    ctrl = dai.CameraControl()
    ctrl.setCaptureStill(True)
    queues_dict[f"control-{mx_id}"].send(ctrl)
    print(f"Sent 'still' event to the camera with MxId: {mx_id}")

def worker(camera, exit_stack, queues_dict):
    device = exit_stack.enter_context(dai.Device(create_pipeline(camera.getMxId()), camera))
    print(f"Using device with MxId: {device.getMxId()}")

    os.makedirs(f"{mission_datetime}/{device.getMxId()}", exist_ok=True)

    queues_dict[f"preview-{device.getMxId()}"] = device.getOutputQueue(name=f"preview-{device.getMxId()}", maxSize=10, blocking=False)
    queues_dict[f"still-{device.getMxId()}"] = device.getOutputQueue(name=f"still-{device.getMxId()}", maxSize=10, blocking=True)
    queues_dict[f"control-{device.getMxId()}"] = device.getInputQueue(name=f"control-{device.getMxId()}")
    queues_dict[f"nn-{device.getMxId()}"] = device.getOutputQueue(name=f"nn-{device.getMxId()}", maxSize=10, blocking=False)
    
    # Wait on the barrier
    barrier.wait()

def gps_thread_function(queues_dict, device_infos):
    # Get list of serial ports
    command = '/bin/bash -c "ls /dev/ttyUSB* /dev/ttyACM*"'
    ports = os.popen(command).read().splitlines()

    # Print ports and prompt user to select one
    for i, port in enumerate(ports):
        print(f"{i+1}. {port}")

    while True:
        try:
            port_num = int(input("Select a port number: ")) - 1
            if port_num < 0 or port_num >= len(ports):
                print("Invalid selection. Please select a number from the list.")
            else:
                selected_port = ports[port_num]
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Prompt user to select baud rate
    while True:
        try:
            baud_rate = int(input("Select a baud rate (9600 or 38400): "))
            if baud_rate not in [9600, 38400]:
                print("Invalid baud rate. Please select either 9600 or 38400.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Connect to GPS
    uart = serial.Serial(selected_port, baudrate=baud_rate, timeout=10)
    gps = GPS(uart, debug=False)

    # Optimize GPS message pipeline - don't send unnecessary data
    gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
    time.sleep(1)
    # Update frequency to highest rate: 10Hz
    gps.send_command(b"PMTK220,100")

    # Define the capture_distance
    capture_distance = 4  # meters

    # Create a CSV file in the new folder and write the header
    with open(os.path.join(mission_datetime, 'gps_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trigger Number", "UTC Time", "Latitude", "Longitude", "Altitude", "HDOP", "Num Sats"])

    # Initialize the trigger counter
    trigger_counter = 0

    # Wait for a valid GPS fix
    while not gps.has_fix:
        gps.update()
        print("No Fix")
        time.sleep(5)

    # Wait for the user to press a key to start recording
    input("Press Enter to start recording...")

    # GPS main loop
    while not gps_quit_flag:
        gps.update()
        if not gps.has_fix:
            print("No Fix")
            time.sleep(5)
            continue

        current_coordinates = (gps.latitude, gps.longitude)

        if trigger_counter == 0:
            print(f"Latitude: {gps.latitude}, Longitude: {gps.longitude}, Altitude: {gps.altitude_m}")
            for device_info in device_infos:
                mx_id = device_info.getMxId()
                ctrl = dai.CameraControl()
                ctrl.setCaptureStill(True)
                queues_dict[f"control-{mx_id}"].send(ctrl)
            print(f"Sent 'still' event to the camera with MxId: {mx_id}")
            last_coordinates = current_coordinates
            trigger_counter += 1
            # Write the initial coordinates to the CSV file
            with open(os.path.join(mission_datetime, 'gps_data.csv'), 'a', newline='') as file:
                utc_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", gps.timestamp_utc)
                writer = csv.writer(file)
                writer.writerow([trigger_counter, utc_time_str, gps.latitude, gps.longitude, gps.altitude_m, gps.hdop, gps.satellites])
        else:
            # Calculate the distance between the last photo and the current position
            displacement = geopy_distance(last_coordinates, current_coordinates).meters
            if displacement > capture_distance:
                print(f"Latitude: {gps.latitude}, Longitude: {gps.longitude}, Altitude: {gps.altitude_m}")
                for device_info in device_infos:
                    mx_id = device_info.getMxId()
                    ctrl = dai.CameraControl()
                    ctrl.setCaptureStill(True)
                    queues_dict[f"control-{mx_id}"].send(ctrl)
                    print(f"Sent 'still' event to the camera with MxId: {mx_id}")
                last_coordinates = current_coordinates
                trigger_counter += 1
                # Write the new coordinates to the CSV file
                with open(os.path.join(mission_datetime, 'gps_data.csv'), 'a', newline='') as file:
                    utc_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", gps.timestamp_utc)
                    writer = csv.writer(file)
                    writer.writerow([trigger_counter, utc_time_str, gps.latitude, gps.longitude, gps.altitude_m, gps.hdop, gps.satellites])
                
        time.sleep(0.1)

        
# Get the date and time of the mission
mission_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
os.makedirs(mission_datetime, exist_ok=True)

device_infos = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')

# Create a barrier for synchronization
barrier = threading.Barrier(len(device_infos) + 1)  # +1 for the main thread

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

    # Wait on the barrier in the main thread
    barrier.wait()

    # Start the GPS thread
    gps_quit_flag = False
    gps_thread = threading.Thread(target=gps_thread_function, args=(queues_dict, device_infos))
    gps_thread.start()

    while True:
        for device_info in device_infos:
            mx_id = device_info.getMxId()

            in_preview = queues_dict[f"preview-{mx_id}"].tryGet()
            if in_preview is not None:
                frame = in_preview.getCvFrame()
            #    cv2.imshow(f"preview-{mx_id}", frame)

            in_neuralnet = queues_dict[f"nn-{mx_id}"].tryGet()
            if in_neuralnet is not None:
                lay1 = np.array(in_neuralnet.getFirstLayerInt32()).reshape(nn_shape,nn_shape)
                found_classes = np.unique(lay1)
                output_colors = decode_deeplabv3p(lay1)
                frame = show_deeplabv3p(output_colors, frame)
                cv2.imshow(f"nn-{mx_id}", frame)

            if queues_dict[f"still-{mx_id}"].has():
                img_frame = queues_dict[f"still-{mx_id}"].get().getData()
                stillimage = cv2.imdecode(np.frombuffer(img_frame, np.uint8), -1)
                counter = counters_dict[mx_id]
                filename = f"{mission_datetime}/{mx_id}/image-{counter}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_frame)
                    print('Image saved to', filename)
                    counters_dict[mx_id] += 1

                # Save NN output as unique jpeg
                if in_neuralnet is not None:  # Check for NN output on given frame
                    fNameDetect = f"{mission_datetime}/{mx_id}/image-detect-{counter}.jpg"
                    # Scale NN output to match full-res RGB image
                    output_colors_resized = cv2.resize(output_colors, nn_jpeg_dims)
                    cv2.imwrite(fNameDetect, output_colors_resized)
                    print('Detection saved to', fNameDetect)
                    # Create a mask from the neural network output
                    ai_mask = np.any(output_colors_resized != [0, 0, 0], axis=-1)
                    # Convert the mask to uint8 and invert it
                    ai_mask = (ai_mask.astype(np.uint8) * 255) ^ 255
                    # Ensure the mask is the same size as the still image
                    ai_mask = cv2.resize(ai_mask, (stillimage.shape[1], stillimage.shape[0]))
                    # Apply the mask to the still image
                    masked_image = cv2.bitwise_and(stillimage, stillimage, mask=ai_mask)
                    # Save the masked image with the "-masked" suffix
                    fNameMasked = f"{mission_datetime}/{mx_id}/image-masked-{counter}.jpg"
                    cv2.imwrite(fNameMasked, masked_image)
                    print('Masked image saved to', fNameMasked)

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