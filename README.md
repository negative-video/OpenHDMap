# Open HD Map

Automatic semantic masking of moving objects with edge-computing to assist Photogrammetry reconstruction, especially for low-cost HD Map creation

## Why?
Self-driving cars will need HD maps to reference themselves within the environment. LIDAR-based HD mapping services are expensive, [alegedly $5,000 USD per kilometer per mission](https://medium.com/syncedreview/the-golden-age-of-hd-mapping-for-autonomous-driving-b2a2ec4c11d), and maps need to be updated frequently to reflect road closures, damage (ie. potholes), and new construction. 

High-resolution cameras are cheap, and [Photogrammetry](https://alicevision.org/#photogrammetry) is a matured process. Photogrammetry is used to reconstruct 3D environments from overlapping imagery taken around an object or through an environment. To save resources, a set number of "features" from each photo are detected, and are compared to all other photos. If a feature is present across photos, calculations are made to determine where these features are in relation to each other. When all relations have been made and intermediate steps are complete, you are left with a 3D point cloud and/or 3D mesh of the subject/environment.

This is problematic for HD Mapping because **we cannot magically determine which features should be detected or ignored.** Moving objects within the scene like cars/trucks/buses driving alongside you or pedestrians/cyclists crossing the street in front of you will confuse the feature-matching process. Because these objects are moving in different trajectories compared to the static earth or yourself, these features will not correlate with those of fixed objects like buildings or bridges. This will cause the photogrammetry algorithm to think the photo was taken in a different perspective from reality, or will otherwise discard the image for low confidence and failure to align. 

Collected imagery [can be masked](https://agisoft.freshdesk.com/support/solutions/articles/31000153479-working-with-masks) to tell the algorithm to avoid features in regions of each photo, but often the process is manual and very tedious. 

**OpenHDMap** was made to automatically mask out these moving objects and provide a pipeline to automatically capture/process HD Mapping imagery for photogrammetry. Utilizing OpenCV AI Kit cameras with an onboard Movidius Myriad X coprocessor, all AI inferencing is done on-camera and allows for a small computer like an Intel NUC to handle 6+ cameras simultaneously. The initial goal was to submit to [SIGSPATIAL,](https://sigspatial2023.sigspatial.org/) but life events got in the way of being ready for the deadline. 

My goal is to create a full mapping solution [less than the cost of a single lidar module](https://www.geoweeknews.com/news/velodyne-cuts-vlp-16-lidar-price-4k), providing similar accuracy for a crowd-sourced network of HD maps for all. I was inspired by George Hotz' blog post [HD Maps for the Masses](https://comma-ai.medium.com/hd-maps-for-the-masses-9a0d582dd274). There hasn't been any development since December of 2018, and I don't want the flame to die out.

The actual "HD Map" generation pipeline is not yet ready. Imagery can be manually added into a Photogrammetry application like Agisoft Metashape, Alicevision Meshroom, or COLMAP.

## How it works:
1. Each OAK camera connected to the computer is booted, initialized, and has its' data packets time-synchronized to ensure every photo captured across multiple cameras happens at the same time
2. For each camera plugged-in and detected, a folder with the name of the camera's unique MxId is created. This is where that camera's images will be saved to.
3. The GPS module, if selected, waits for accurate position
4. Camera pipelines are initialized with a semantic segmentation model (currently DeepLabV3+) working to detect [any of the 20 PASCAL VOC classes](http://host.robots.ox.ac.uk/pascal/VOC/) and highlighting them within the camera preview window
5. When a photo is triggered, whether manually, at time interval, or by GPS, the following happens:
   
   a. A full-resolution, non-masked image is saved as `image-####.jpg` in its' respective MxId folder, where `####` represents the incremental count of photos in the mission
   
   b. For that same frame, output of the OAK's neural net pipeline is stretched from its native 256x256 pixels to match the camera dimensions and is saved as `image-OAKdetect-####.jpg`, where `####` matches the same count as above. This photo is black except where the model detected classes, with each color representing a different class

   c. The two above images are laid on top of each other. Any "colored" pixels in `image-OAKdetect-####.jpg` are replaced with black pixels. This composite is saved as a third file, `image-OAKmasked-####.jpg` where, once again, `####` matches the other two saved images. It will look identical to the first image except where cars/trucks/pedestrians/etc are blacked out.

## Planned additional steps:
1. Provide script to mask out images not captured by OAK cameras
2. Lens interinsics and rig extrinsics can be computed and saved somewhere to speed up processing
3. Every X distance or X incremental captures, a photogrammetry application script will automatically import masked photos and process to generate a low/medium-quality point cloud for this section of road
4. Using a cellular data connection to the computer, these point clouds will be copied to a remote server using a combination of Tailscale SD-WAN and Rsync 

## How to test code:
* Follow Luxonis' [guide for installing DepthAI on your computer](https://docs.luxonis.com/projects/api/en/latest/install/)
* Install Python 3.X and venv if not already done
* [Create a virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/) and activate it
* Download this repo and navigate to its' directory
* Run the command `pip install -r requirements.txt`
* Physically connect OAK camera to computer with USB-3 cable
* Run `multicam-manual-capture-masking.py` for the demo utilizing onboard AI masking, or `multicam-manual-photo.py` to just take synchronized, high-resolution photos
* Assuming the camera connects and initializes, a preview window will load showing camera output (overlayed with neural net detections, if you ran `multicam-manual-capture-masking.py`). Mouse-click this preview window to ensure keyboard capture
* Press the `c` key on your keyboard to take a photo. They will be saved to a folder matching the camera's MxId.
* When done, press the `q` key to quit the program


## How to recreate mapping rig
* Follow above installation steps
* ToDo

## Important notes:
* Only the RGB camera on each OAK is used, **not the stereo depth cameras.** I used OAK-D-Lite cameras because they were on-hand. Save your money and do not buy OAK depth cameras specifically for this project!

* Cameras with an IMX378 sensor for RGB photos take better-quality pictures than those with IMX214 sensors. Consider the OAK-1 over OAK-1-Lite if at all possible.

* **True capture delay is unknown**. With 10Hz position updates on the Adafruit Ultimate GPS USB module and 25ms polling of the serial port, the worst-case positional time disparity is 125ms. Delay between triggering the capture and recieving the frame (not saving to disk) averages 135ms. *I don't know when the frame is pulled for saving, once a capture is triggered.* This could mean anywhere from 10ms -> 260ms could have elapsed since a location was obtained and the image taken. For a vehicle traveling at speed while taking photos, this drift should not be ignored:

  | Speed  | Time Disparity | Distance Traveled |
  | ------ | -------------- | ----------------- |
  | 25mph/40.2kph  | 10ms | 0.36ft/0.11m |
  | 25mph/40.2kph  |  260ms | 9.53ft/2.9m |
  | 60mph/96.6kph | 10ms | 0.88ft/0.27m |
  | 60mph/96.6kph | 260ms | 22.88ft/6.907m |
  


* These scripts assume you are using an OAK module with a 12MP sensor, such as OAK-D, OAK-D-Lite, or OAK-1. If your sensor is not natively 12MP, be sure to adjust `THE_12_MP` within the class `ColorCameraProperties.SensorResolution` according to [Luxonis' supported resolutions](https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ColorCameraProperties.SensorResolution) to the maximum supported by your camera on its' product page.

* Current AI detections run continuously. After fully testing pipeline, it would be preferrable to lower power consumption/heat output and only pass triggered snapshot preview to AI model, such as in Luxonis' script for [script_change_pipeline_flow](https://docs.luxonis.com/projects/api/en/latest/samples/Script/script_change_pipeline_flow/)

* FDM/FFF 3D printing for the construction of the camera rig is not advised. Each layer is a weak point that could break due to vibrations or bumps. Material needs to be considered for its thermal properties as well.

* It's probably not the best idea to conduct mapping in a heat wave or with precipitation. Needless to say, none of the hardware here is weatherproof.


## ToDo
* Get car fixed to test on real-world environments
* Recreate steps to make mapping rig
* Wireless serial pipe with ESP-NOW for driver dashboard status 
* Test on lower-power computer hardware (Pi Zero 2W)
* Append metadata to each saved photo (location, lens distortion)
* Automate Photogrammetry point-cloud creation
* Add IMU for more accurate positioning
* Optimize AI inferencing to only run on snapshots
* Test RTK module with public corrections data
* Conduct testing with [Luxonis' latency measurement](https://docs.luxonis.com/projects/api/en/latest/samples/host_side/latency_measurement/)