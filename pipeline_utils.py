import depthai as dai
import os
import config

def create_pipeline(mx_id):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(config.nn_path)
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
    cam.setPreviewSize(config.nn_shape,config.nn_shape)
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
    config.process_next_frame
    mx_id = device_info.getMxId()
    ctrl = dai.CameraControl()

    # Pull out of Auto mode
    ctrl.setManualExposure(500, 1000)  # 500 microseconds exposure and ISO 600
    ctrl.setManualFocus(123)  # "Infinity" on OAK-D

    ctrl.setCaptureStill(True)
    queues_dict[f"control-{mx_id}"].send(ctrl)
    print(f"Sent 'still' event to the camera with MxId: {mx_id}")
    config.process_next_frame = True

def worker(camera, exit_stack, queues_dict):
    device = exit_stack.enter_context(dai.Device(create_pipeline(camera.getMxId()), camera))
    print(f"Using device with MxId: {device.getMxId()}")

    os.makedirs(f"captures/{config.mission_datetime}/{device.getMxId()}", exist_ok=True)

    queues_dict[f"preview-{device.getMxId()}"] = device.getOutputQueue(name=f"preview-{device.getMxId()}", maxSize=10, blocking=False)
    queues_dict[f"still-{device.getMxId()}"] = device.getOutputQueue(name=f"still-{device.getMxId()}", maxSize=10, blocking=True)
    queues_dict[f"control-{device.getMxId()}"] = device.getInputQueue(name=f"control-{device.getMxId()}")
    queues_dict[f"nn-{device.getMxId()}"] = device.getOutputQueue(name=f"nn-{device.getMxId()}", maxSize=10, blocking=False)
    
    # Wait on the barrier
    config.barrier.wait()