import os
import shutil
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np

# Prefer GPU if present
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.to(device)
model.eval()

# Define the image transformations
transform = T.Compose([
    T.ToTensor(),
    # Standard values for mean/std for ImageNet dataset photos
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
])

# PASCAL-VOC classes used by DeepLabV3
background = (0, 0, 0)
aeroplane = (128, 0, 0)
bicycle = (0, 128, 0)
bird = (128, 128, 0)
boat = (0, 0, 128)
bottle = (128, 0, 128)
bus = (0, 128, 128)
car = (128, 128, 128)
cat = (64, 0, 0)
chair = (192, 0, 0)
cow = (64, 128, 0)
dining_table = (192, 128, 0)
dog = (64, 0, 128)
horse = (192, 0, 128)
motorbike = (64, 128, 128)
person = (192, 128, 128)
potted_plant = (0, 64, 0)
sheep = (128, 64, 0)
sofa = (0, 192, 0)
train = (128, 192, 0)
tv_monitor = (0, 64, 128)

# Build array for inferencing
label_map = np.array([background,aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,dining_table,
                      dog,horse,motorbike,person,potted_plant,sheep,sofa,train,tv_monitor,])

# Index of above things that move (we want to mask out)
masking_classes = {1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19}

input_dir = 'input_imagery'
output_dir = 'output_imagery'
os.makedirs(output_dir, exist_ok=True)

# Loop to process all jpegs in folder
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        # Perform semantic segmentation
        with torch.no_grad():
            output = model(input_image)['out'][0]
        # Numpy ops cannot use GPU, so move to CPU
        output_predictions = output.argmax(0).cpu().numpy()

        # Discard detections of irrelevant classes
        output_predictions = np.where(np.isin(output_predictions, list(masking_classes)), output_predictions, 0)

        # Color detections
        output_color = label_map[output_predictions]

        # Detection output
        output_image = Image.fromarray(output_color.astype('uint8'))
        output_image = output_image.resize((image.width, image.height), Image.NEAREST)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '-Torchdetect.jpg')
        output_image.save(output_path)

        # Mask
        mask = np.all(output_color == background, axis=-1)
        input_image_np = np.array(image)
        masked_image = np.where(mask[..., None], input_image_np, 0)
        masked_image = Image.fromarray(masked_image)
        masked_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '-Torchmasked.jpg')
        masked_image.save(masked_path)

        # Move the original image to the output directory
        shutil.move(image_path, os.path.join(output_dir, filename))
        print("Processed " + filename)