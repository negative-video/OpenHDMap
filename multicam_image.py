import datetime
from pathlib import Path
import cv2
import depthai as dai
import contextlib

def createPipeline():
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setPreviewSize(420, 312)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # Max = THE_13_MP or THE_4_k or THE_5_MP

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)

    xin = pipeline.create(dai.node.XLinkIn)
    xin.setStreamName("control")
    xin.out.link(camRgb.inputControl)
    xin.setNumFrames(2) # Frame buffer occupies all RAM at high resolution; drop from 4 to 2

    # Properties
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
    camRgb.still.link(videoEnc.input)

    # Linking
    xoutStill = pipeline.create(dai.node.XLinkOut)
    xoutStill.setStreamName("still")
    videoEnc.bitstream.link(xoutStill.input)

    return pipeline


with contextlib.ExitStack() as stack:
    # Record from all available devices
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0:
         raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")

    devices = []
    qRgbMap = []
    for device_info in device_infos:
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2Mode=False))
        mxId = device.getMxId()
        cameras = device.getConnectedCameras()
        usbSpeed = device.getUsbSpeed()
        eepromData = device.readCalibration2().getEepromData()

        pipeline = createPipeline()
        device.startPipeline(pipeline)

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
        qStill = device.getOutputQueue(name="still", maxSize=30, blocking=True)
        qControl = device.getInputQueue(name="control")
        stream_name = "rgb-" + mxId + "-" + eepromData.productName
        qRgbMap.append((qRgb, stream_name))

    # Make sure the destination path is present before starting to store the examples
    dirName = "rgb_data"
    Path(dirName).mkdir(parents=True, exist_ok=True)
 
    while True:
        for qRgb, stream_name in qRgbMap:
            if qRgb.has():
                frame = qRgb.get().getCvFrame()
                frame = cv2.pyrDown(frame)
                frame = cv2.pyrDown(frame)
                frame = cv2.pyrDown(frame)
                cv2.imshow(stream_name, frame)



#            inRgb = qRgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise
#            if inRgb is not None:
#                frame = inRgb.getCvFrame()
                # 4k / 8
#                frame = cv2.pyrDown(frame)
#                frame = cv2.pyrDown(frame)
#                frame = cv2.pyrDown(frame)
#                cv2.imshow("rgb", frame)

        if qStill.has():
            now = datetime.datetime.now()
            date_time_file = now.strftime("%Y-%m-%d-%H-%M-%S")
            with open(dirName + "/" + date_time_file + "--" + stream_name + ".jpeg", "wb") as f:
                f.write(qStill.get().getData())
                print('Image saved to', date_time_file)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            qControl.send(ctrl)
            print("Sent 'still' event to the camera!")
