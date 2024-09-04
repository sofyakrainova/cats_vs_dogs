import shutil
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

base_dir = "../Kaggle_data/cats_vs_dogs/train/"
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


print("Creating train and validation folders")
try:
  create_train_val_dirs(root_path=base_dir)
except FileExistsError:
  print("Directories are already created")

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets

  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training

  Returns:
    None
  """
  all_files = os.listdir(SOURCE_DIR)
  good_files = []
  for f in all_files:
    if os.path.getsize(os.path.join(SOURCE_DIR, f))>0:
      good_files.append(f)
    else:
      print(f+" filename is zero length, so ignoring.")
  source_size = len(good_files)
  train_files = random.sample(good_files, int(source_size*SPLIT_SIZE))
  for f in good_files:
    if f in train_files:
      shutil.copyfile(os.path.join(SOURCE_DIR, f), os.path.join(TRAINING_DIR, f))
    else:
      shutil.copyfile(os.path.join(SOURCE_DIR, f), os.path.join(VALIDATION_DIR, f))

TRAINING_DIR = os.path.join(base_dir, "training/")
VALIDATION_DIR = os.path.join(base_dir, "validation/")

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")

# Define proportion of images used for training
split_size = .9
print("Split datasets, takes som ")

split_data(cats_dir, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(dogs_dir, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

# Training and validation splits
print(f"There are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")