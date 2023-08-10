import cv2
import config
import numpy as np
from deeplabv3p_utils import decode_deeplabv3p, show_deeplabv3p

def run_camera_loop(device_infos, queues_dict, counters_dict):
    for device_info in device_infos:
        mx_id = device_info.getMxId()

        # Only process and display the frame if the flag is set
        if config.process_next_frame:

            in_preview = queues_dict[f"preview-{mx_id}"].tryGet()
            if in_preview is not None:
                frame = in_preview.getCvFrame()
            #    cv2.imshow(f"preview-{mx_id}", frame)

            in_neuralnet = queues_dict[f"nn-{mx_id}"].tryGet()
            if in_neuralnet is not None:
                lay1 = np.array(in_neuralnet.getFirstLayerInt32()).reshape(config.nn_shape,config.nn_shape)
                found_classes = np.unique(lay1)
                output_colors = decode_deeplabv3p(lay1)
                frame = show_deeplabv3p(output_colors, frame)
                cv2.imshow(f"nn-{mx_id}", frame)

            if queues_dict[f"still-{mx_id}"].has():
                img_frame = queues_dict[f"still-{mx_id}"].get().getData()
                stillimage = cv2.imdecode(np.frombuffer(img_frame, np.uint8), -1)
                counter = counters_dict[mx_id]
                filename = f"captures/{config.mission_datetime}/{mx_id}/image-{counter}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_frame)
                    print('Image saved to', filename)
                    counters_dict[mx_id] += 1

                # Save NN output as unique jpeg
                if in_neuralnet is not None:  # Check for NN output on given frame
                    fNameDetect = f"captures/{config.mission_datetime}/{mx_id}/image-detect-{counter}.jpg"
                    # Scale NN output to match full-res RGB image
                    output_colors_resized = cv2.resize(output_colors, config.nn_jpeg_dims)
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
                    fNameMasked = f"captures/{config.mission_datetime}/{mx_id}/image-masked-{counter}.jpg"
                    cv2.imwrite(fNameMasked, masked_image)
                    print('Masked image saved to', fNameMasked)