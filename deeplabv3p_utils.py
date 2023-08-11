import numpy as np
import cv2
import config

def decode_deeplabv3p(output_tensor):
    output = output_tensor.reshape(config.nn_shape,config.nn_shape)
    output = np.array(output) * (255/config.num_of_classes)
    output = output.astype(np.uint8)
    output_colors = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    output_colors[output == 0] = [0,0,0]
    return output_colors

def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame,1, output_colors,0.4,0)