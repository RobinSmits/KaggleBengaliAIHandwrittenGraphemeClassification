# Kaggle Bengali.AI Handwritten Grapheme Classification

In this repository you can find some of the code I use for the Kaggle Bengali.AI competition. 

## Kernel
In the folder 'KaggleKernelEfficientNetB3' you can find the part of the code I used to train the models used in my inference [kernel](https://www.kaggle.com/rsmits/keras-efficientnet-b3-training-inference) as posted on Kaggle.

To be able to train the model you need to first download the dataset as available for the [Kaggle competition](https://www.kaggle.com/c/bengaliai-cv19).

The model consists of an EfficientNet B3 pre-trained (on Imagenet..) model with a generalized mean pool and custom head layer.
For image preprocessing I just invert, normalize and scale the image ... nothing else. No form of augmentation is used.

The code should be able to run on any Python 3.6/3.7 environment. Major packages used were:
- Tensorflow 2.1.0
- Keras 2.3.1
- Efficientnet 1.0.0
- Opencv-python
- Iterative-stratification 0.1.6

I've trained the model for 80 epochs and picked some model weight files todo an ensemble in the inference kernel. 

I first tested the training part by using 5/6th of the training data for training and 1/6th for validation. Based on the validation and some leaderboard submissions I found that the highest scoring epochs were between epoch 60 - 70. In the final training (as is used in this code) I use a different distribution of the training data for every epoch. The downside of this is that validation doesn't tell you everything anymore. The major benefit is however that it increases the score about 0.005 to 0.008 compared to the use of the fixed training set. This way it get close to what a Cross Validation ensemble would do...just without the training needed for that.

The model weight files as used in the inference kernel are available in folder 'KaggleKernelEfficientNetB3\model_weights'. It contains the following 4 files (for each file mentioned the LB score when used as a single file to generate the submission):
- Train1_model_59.h5     LB 0.9681
- Train1_model_64.h5     LB 0.9679
- Train1_model_66.h5     LB 0.9685
- Train1_model_68.h5     LB 0.9691

To start training the model you need to use the train.py file and at least verify/modify the following values:
- DATA_DIR      (this should be the directory with the Bengali.AI dataset)
- TRAIN_DIR     (this should be the directory where you want to store the generated training images) 
- GENERATE_IMAGES   (whether or not the training images must be pre-generated..should be done initially)