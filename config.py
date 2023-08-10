from datetime import datetime
import threading

# Neural Net Parameters
nn_path = 'ai_models/deeplab_v3_plus_mnv2_decoder_256_openvino_2021.4.blob' # path to AI model
gps_quit_flag = False
nn_shape = 256
nn_jpeg_dims = (2016,1520) # Needs to be rewritten to account for scaling up/down
num_of_classes = 21 # define the number of classes in the dataset

mission_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

gps_quit_flag = False

# Global flag to process and display next frame in preview
process_next_frame = False

# Create a barrier function for camera sync
barrier = None
def set_barrier(device_count):
    global barrier
    barrier = threading.Barrier(device_count + 1) # +1 for the main thread