# cats_vs_dogs
project for image recognition 2 classes

This project uses data from Kaggle datasets, which can be loaded here https://www.kaggle.com/competitions/dog-vs-cat-classification/data  
For training there are 25.000 images which are separated in the two folders dog and cat relevant to each class.  
For test there are 8,000 images without labels.  
  
For this project I will use different CNN and image augmentstions.

============== Baseline model ================================  
Model with 3 Conv2D layers and 2 Dense layers, target_size = 150, 150, no image augmentation.  
After parameters selection we have:  
First layer 8 (3,3), second layer 16 (3,3), third layer 28 (3,3), dense layer 320  
After 20 epochs we have clear overfitting during training:  
Train accuracy 0.998  
Validation accuracy 0.837  
Test accuracy 0.984
   
============= Image augmentation model =========================  
After adding augmentation for baseline model it is clearly underfitted, it needs more epochs to train and more complicated model.  
After parameter selection we have:  
First layer 26 (3,3), second layer 20 (4,4), third layer 60 (3,3), dense layer 32. After 30 epochs we have, we need 50 at least.  
Train accuracy  0.852   
Validation accuracy 0.844  
Test accuracy 0.859  
  
============ Transfer Learning ===================================  
Here we will retrain existing model for image classification. Weights for the model can be downloaded from here:  
https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  
After parameters tuning we have dense layer 256 and dense layer 576.  With image size 300x300 and epochs 7 we have next results:  
Train accuracy 0.999  
Validation accuracy 0.996  
Test accuracy: 0.999  
