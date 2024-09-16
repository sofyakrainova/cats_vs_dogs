from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

TEST_DIR = "../Kaggle_data/cats_vs_dogs/"
TARGET_SIZE = 300
savedModel=load_model("trained_model_dropout.h5")
print(savedModel.summary())

test_datagen = ImageDataGenerator(rescale=1/255.)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                              classes=['test'],
                              class_mode=None,
                              shuffle=False,
                              target_size=(TARGET_SIZE, TARGET_SIZE),
                              batch_size= 1)

preds = savedModel.predict(test_generator).flatten().astype(int)

filenames=test_generator.filenames

results=pd.DataFrame({"id":filenames,
                      "labels":preds})

results["id"] = results["id"].apply(lambda x: x[5:])
results.to_csv("submission4.csv",index=False)


