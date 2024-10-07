from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TARGET_SIZE = 300
EPOCHS = 7
TRAINING_DIR = "../Kaggle_data/cats_vs_dogs/train/training"
VALIDATION_DIR = "../Kaggle_data/cats_vs_dogs/train/validation"
local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators

  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images

  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  train_datagen = ImageDataGenerator( rescale = 1.0/255,

                                      )
  train_generator = train_datagen.flow_from_directory(
                                                      directory=TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode="binary",
                                                      target_size=(TARGET_SIZE, TARGET_SIZE)
                                                      )

  validation_datagen = ImageDataGenerator( rescale = 1.0/255. )
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode="binary",
                                                                target_size=(TARGET_SIZE, TARGET_SIZE))
  return train_generator, validation_generator

train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)

# Loading pretrained model
pre_trained_model = InceptionV3(input_shape = (TARGET_SIZE, TARGET_SIZE, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)
# Making sure trained layers will stay intact
for layer in pre_trained_model.layers:
  layer.trainable = False

# Last trained layer we will use
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# Add new layers to the model
x = layers.Flatten()(last_output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(576, activation='relu')(x)
x = layers.Dense  (1, activation='sigmoid')(x)

# Append the dense network to the base model
model = Model(pre_trained_model.input, x)

model.summary()

model.compile(optimizer=RMSprop(learning_rate=1e-05),
                loss="binary_crossentropy",
                metrics=["accuracy"])

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            epochs = EPOCHS,
            verbose = 2)

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', label = "Training Accuracy")
plt.plot(epochs, val_acc, 'b', label = "Validation Accuracy")
plt.grid()
plt.legend()
plt.title('Training and validation accuracy')
plt.show()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', label = "Training Loss")
plt.plot(epochs, val_loss, 'b', label = "Validation Loss")
plt.legend()
plt.grid()
plt.show()

# Save the weights
model.save('trained_model_dropout.h5')