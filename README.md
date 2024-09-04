# cats_vs_dogs
project for image recognition 2 classes

This project uses data from Kaggle datasets, which can be loaded here https://www.kaggle.com/competitions/dog-vs-cat-classification/data
For training there are 25.000 images which are separated in the two folders dog and cat relevant to each class.
For test there are 8,000 images without labels.

For this project I will use different CNN and image augmentstions.

============== Baseline model =======================
Model with 3 Conv2D layers and 2 Dense layers, target_size = 150, 150, no image augmentation
Clear overfitting during training:
Train accuracy > 0.99
Validation accuracy 0.83

============= Image augmentation model =========================

