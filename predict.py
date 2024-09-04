from keras.src.utils.module_utils import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

TEST_DIR = "."

savedModel=load_model("gfgModel.h5")
print(savedModel.summary())

test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                              classes=['test'],
                              # don't generate labels
                              class_mode=None,
                              # don't shuffle
                              shuffle=False,
                              # use same size as in training
                              target_size=(150, 150),
                              batch_size= 1)

preds = savedModel.predict(test_generator).flatten().astype(int)
print(preds)

filenames=test_generator.filenames
results=pd.DataFrame({"id":filenames,
                      "labels":preds})
results.to_csv("submission1.csv",index=False)


