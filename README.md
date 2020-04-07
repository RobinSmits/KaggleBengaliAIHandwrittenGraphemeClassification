# Kaggle Bengali.AI Handwritten Grapheme Classification

In this repository you can find some of the code I use for the Kaggle Bengali.AI competition. 

## Kernel
In the folder 'KaggleKernelEfficientNetB3' you can find the part of the code I used to train the models used in my inference [kernel](https://www.kaggle.com/rsmits/keras-efficientnet-b3-training-inference) as posted on Kaggle.

The model scored 0.9703 on the Public Board and 0.9182 on the Private Board

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

The model weight files as used in the inference kernel are available in folder 'KaggleKernelEfficientNetB3\model_weights'. It contains the following 6 files (for each file mentioned the LB score when used as a single file to generate the submission):
- Train1_model_57.h5     Public LB: 0.9668  -  Private LB: 0.9132
- Train1_model_59.h5     Public LB: 0.9681  -  Private LB: 0.9151
- Train1_model_64.h5     Public LB: 0.9679  -  Private LB: 0.9167
- Train1_model_66.h5     Public LB: 0.9685  -  Private LB: 0.9157
- Train1_model_68.h5     Public LB: 0.9691  -  Private LB: 0.9167
- Train1_model_70.h5     Public LB: 0.9700  -  Private LB: 0.9174

To start training the model you need to use the train.py file and at least verify/modify the following values:
- DATA_DIR      (this should be the directory with the Bengali.AI dataset)
- TRAIN_DIR     (this should be the directory where you want to store the generated training images) 
- GENERATE_IMAGES   (whether or not the training images must be pre-generated..should be done initially)

To run the inference on the models you can use the Jupyter Notebook 'keras-efficientnet-b3-training-inference.ipynb'. Note that you do need to modify some paths to the model weight files.

## Competition Final Submission
In the folder 'KaggleFinalSubmission' you can find the part of the code I used together with my Kaggle team to train the models used our final submission(s). In the final submissions we also used some SE-ResNeXt models in an ensemble. The same guidelines apply to this model as mentioned above.

Just the EfficientNet B3 model however already gave the same score as the multi-model ensemble. The model scored 0.9739 on the Public Board and 0.9393 on the Private Board. 

With our submission we achieved a 47th place out of 2000+ participants.

Use the final submission inference kernel and the provided model weights to try it out. To be able to train the model you need to first download the dataset as available for the [Kaggle competition](https://www.kaggle.com/c/bengaliai-cv19).