# Open HD Map
### Gerald Burkett

Goal: 
 Allow anyone to HD maps of environment for low cost

Objectives:
* Synchronize multiple OpenCV AI Kit cameras
* Collect imagery from cameras at set intervals
* Collect GPS data and append to collected images
* Allow IMU data to be saved for VIO or VSLAM
* Collect data + train model on pedestrian/vehicle/construction imagery
* Use Myriad X in OAK sensors to "white out" this data, leaving only envoronment in scene
* Import masked images into Metashape or COLMAP for reconstruction
* Pipeline auto import, processing and export
* Determine ideal image interval and parameters for point cloud export
* Optimize to run on more hardware

## Contains the following files:
gps_only.py - logs gps attribues to .csv file for later appending 