import shutil
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

base_dir = "all_data/"
dogs_dir = os.path.join(base_dir, "dogs/")
cats_dir = os.path.join(base_dir, "cats/")

# Print sample images
plt.imshow(load_img(f"{os.path.join(dogs_dir, os.listdir(dogs_dir)[0])}"))
plt.show()
plt.imshow(load_img(f"{os.path.join(cats_dir, os.listdir(cats_dir)[0])}"))
plt.show()

# check images shape and maximum value for pixels
sample_image  = load_img(f"{os.path.join(dogs_dir, os.listdir(dogs_dir)[0])}")
# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)
print(f"Each image has shape: {sample_array.shape}")
print(f"The maximum pixel value used is: {np.max(sample_array)}")

def create_train_val_dirs(root_path):

  train_dir = os.path.join(root_path, 'training')
  validation_dir = os.path.join(root_path, 'validation')
  os.makedirs(train_dir)
  os.makedirs(validation_dir)

  # Directory with training cat/dog pictures
  train_cats_dir = os.path.join(train_dir, 'cats')
  train_dogs_dir = os.path.join(train_dir, 'dogs')
  os.makedirs(train_cats_dir)
  os.makedirs(train_dogs_dir)

  # Directory with validation cat/dog pictures
  validation_cats_dir = os.path.join(validation_dir, 'cats')
  validation_dogs_dir = os.path.join(validation_dir, 'dogs')
  os.makedirs(validation_cats_dir)
  os.makedirs(validation_dogs_dir)

try:
  create_train_val_dirs(root_path=base_dir)
except FileExistsError:
  print("You should not be seeing this since the upper directory is removed beforehand")

