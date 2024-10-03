from keras.src.utils.module_utils import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

TEST_DIR = "../Kaggle_data/cats_vs_dogs/"
TARGET_SIZE = 150
savedModel=load_model("trained_model.h5")
print(savedModel.summary())

test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                              classes=['test'],
                              # don't generate labels
                              class_mode=None,
                              # don't shuffle
                              shuffle=False,
                              # use same size as in training
                              target_size=(TARGET_SIZE, TARGET_SIZE),
                              batch_size= 1)

preds = savedModel.predict(test_generator).flatten().astype(int)

filenames=test_generator.filenames

results=pd.DataFrame({"id":filenames,
                      "labels":preds})

results["id"] = results["id"].apply(lambda x: x[5:])
results.to_csv("submission.csv",index=False)


