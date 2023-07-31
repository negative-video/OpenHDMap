import cv2
import depthai as dai
import threading
import contextlib
import time
import os

def create_pipeline(mx_id):
    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

    # Set up a preview link with a lower resolution
    previewOut = pipeline.create(dai.node.XLinkOut)
    previewOut.setStreamName(f"preview-{mx_id}")
    camRgb.preview.link(previewOut.input)

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName(f"control-{mx_id}")
    xin.out.link(camRgb.inputControl)

    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    camRgb.still.link(videoEnc.input)

    xoutStill = pipeline.create(dai.node.XLinkOut)
    xoutStill.setStreamName(f"still-{mx_id}")
    videoEnc.bitstream.link(xoutStill.input)

    return pipeline

def worker(device_info, exit_stack, queues_dict):
    device = exit_stack.enter_context(dai.Device(create_pipeline(device_info.getMxId()), device_info))
    print(f"Using device with MxId: {device.getMxId()}")

    os.makedirs(device.getMxId(), exist_ok=True)

    queues_dict[f"preview-{device.getMxId()}"] = device.getOutputQueue(name=f"preview-{device.getMxId()}", maxSize=30, blocking=False)
    queues_dict[f"still-{device.getMxId()}"] = device.getOutputQueue(name=f"still-{device.getMxId()}", maxSize=30, blocking=True)
    queues_dict[f"control-{device.getMxId()}"] = device.getInputQueue(name=f"control-{device.getMxId()}")


device_infos = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')

with contextlib.ExitStack() as exit_stack:
    queues_dict = {}
    threads = []

    for device_info in device_infos:
        time.sleep(1) # Currently required due to XLink race issues
        thread = threading.Thread(target=worker, args=(device_info, exit_stack, queues_dict))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join() # Wait for all threads to finish

    counters = {device_info.getMxId(): 0 for device_info in device_infos}
    counters_dict = {}
    for device_info in device_infos:
        counters_dict[device_info.getMxId()] = 0

    while True:
        for device_info in device_infos:
            mx_id = device_info.getMxId()

            in_preview = queues_dict[f"preview-{mx_id}"].tryGet()  
            if in_preview is not None:
                frame = in_preview.getCvFrame()
                cv2.imshow(f"preview-{mx_id}", frame)

            if queues_dict[f"still-{mx_id}"].has():
                img_frame = queues_dict[f"still-{mx_id}"].get().getData()
                counter = counters_dict[mx_id]
                filename = f"{mx_id}/image{counter}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_frame)
                    print('Image saved to', filename)
                    counters_dict[mx_id] += 1

        # Handle key presses here, after updating all frames
        key = cv2.waitKey(1)
        if key == ord('c'):
            for device_info in device_infos:
                mx_id = device_info.getMxId()
                ctrl = dai.CameraControl()
                ctrl.setCaptureStill(True)
                queues_dict[f"control-{mx_id}"].send(ctrl)
                print(f"Sent 'still' event to the camera with MxId: {mx_id}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

print('Devices closed')