#!/usr/bin/env python3
import os
import cv2
import time
import threading
import contextlib
import numpy as np
import depthai as dai

# Neural Net Parameters
nn_path = 'ai_models/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob' # path to AI model
num_of_classes = 21 # define the number of classes in the dataset
nn_shape = 256
nn_jpeg_dims = (2016,1520) # Needs to be rewritten to account for scaling up/down

### Start Function Blocks
def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(nn_shape,nn_shape)
    output = np.array(output) * (255/num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    output_colors[output == 0] = [0,0,0]
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.4,0)

def create_pipeline(mx_id): # Initializes every connected camera
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

def worker(camera, exit_stack, queues_dict):
    device = exit_stack.enter_context(dai.Device(create_pipeline(camera.getMxId()), camera))
    print(f"Using device with MxId: {device.getMxId()}")

    os.makedirs(device.getMxId(), exist_ok=True)

    queues_dict[f"preview-{device.getMxId()}"] = device.getOutputQueue(name=f"preview-{device.getMxId()}", maxSize=10, blocking=False)
    queues_dict[f"still-{device.getMxId()}"] = device.getOutputQueue(name=f"still-{device.getMxId()}", maxSize=10, blocking=True)
    queues_dict[f"control-{device.getMxId()}"] = device.getInputQueue(name=f"control-{device.getMxId()}")
    queues_dict[f"nn-{device.getMxId()}"] = device.getOutputQueue(name=f"nn-{device.getMxId()}", maxSize=10, blocking=False)
### End Function Blocks

all_oak_cameras = dai.Device.getAllAvailableDevices()
print(f'Found {len(all_oak_cameras)} devices')

# Individual processes for EACH OAK camera
with contextlib.ExitStack() as exit_stack:
    queues_dict = {}
    threads = []

    for camera in all_oak_cameras:
        time.sleep(1) # XLink race issues?
        thread = threading.Thread(target=worker, args=(camera, exit_stack, queues_dict))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join() # Wait for all threads to finish

    counters = {camera.getMxId(): 0 for camera in all_oak_cameras}
    counters_dict = {}
    for camera in all_oak_cameras:
        counters_dict[camera.getMxId()] = 0

    while True:
        for camera in all_oak_cameras:
            mx_id = camera.getMxId()

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
                filename = f"{mx_id}/image-{counter}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_frame)
                    print('Image saved to', filename)
                    counters_dict[mx_id] += 1

                # Save NN output as unique jpeg
                if in_neuralnet is not None:  # Check for NN output on given frame
                    fNameDetect = f"{mx_id}/image-detect-{counter}.jpg"
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
                    fNameMasked = f"{mx_id}/image-masked-{counter}.jpg"
                    cv2.imwrite(fNameMasked, masked_image)
                    print('Masked image saved to', fNameMasked)

        # Handle key presses here, after updating all frames
        key = cv2.waitKey(1)
        if key == ord('c'):
            for camera in all_oak_cameras:
                mx_id = camera.getMxId()
                ctrl = dai.CameraControl()
                ctrl.setCaptureStill(True)
                queues_dict[f"control-{mx_id}"].send(ctrl)
                print(f"Sent 'still' event to the camera with MxId: {mx_id}")
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()
print('Devices closed')