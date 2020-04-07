# Import Modules
import os
import time, gc
import numpy as np
import pandas as pd
from math import floor
import cv2
import tensorflow as tf

# Keras
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint

# Iterative-Stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

# Custom 
from preprocessing import generate_images, resize_image
from model import create_model
from utils import plot_summaries

# Seeds
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Input Dir
DATA_DIR = 'C:/KaggleBengaliAI/bengaliai-cv19'
TRAIN_DIR = './train/'

# Constants
HEIGHT = 137
WIDTH = 236
SCALE_FACTOR = 1.00
HEIGHT_NEW = int(HEIGHT * SCALE_FACTOR)
WIDTH_NEW = int(WIDTH * SCALE_FACTOR)
RUN_NAME = 'Train2_'
PLOT_NAME1 = 'Train2_LossAndAccuracy.png'
PLOT_NAME2 = 'Train2_Recall.png'

BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 70
TEST_SIZE = 1./6

# Image Size Summary
print(HEIGHT_NEW)
print(WIDTH_NEW)

# Generate Image (Has to be done only one time .. or again when changing SCALE_FACTOR)
GENERATE_IMAGES = True
if GENERATE_IMAGES:
    generate_images(DATA_DIR, TRAIN_DIR, WIDTH, HEIGHT, WIDTH_NEW, HEIGHT_NEW)

# Prepare Train Labels (Y)
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
tgt_cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
desc_df = train_df[tgt_cols].astype('str').describe()
types = desc_df.loc['unique',:]
X_train = train_df['image_id'].values
train_df = train_df[tgt_cols].astype('uint8')
for col in tgt_cols:
    train_df[col] = train_df[col].map('{:03}'.format)
Y_train = pd.get_dummies(train_df)

# Cleanup
del train_df
gc.collect()

# Modelcheckpoint
def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name, 
                            monitor = 'val_loss', 
                            verbose = 1, 
                            save_best_only = False, 
                            save_weights_only = True, 
                            mode = 'min', 
                            period = 1)

def _read(path):
    img = cv2.imread(path)    
    return img

# Mixup implementation Modified for this competition.
def mixup(alpha, x, y_root, y_vowel, y_consonant):
    n = x.shape[0]
    l = np.random.beta(alpha, alpha, n)
    x_l = l.reshape(n, 1, 1, 1)
    y_root_l = l.reshape(n, 1)
    y_vowel_l = l.reshape(n, 1)
    y_consonant_l = l.reshape(n, 1)

    # X
    x1 = x
    x2 = np.flip(x, axis=0)
    x = x1 * x_l + x2 * (1. - x_l)

    # y_root
    yroot1 = y_root
    yroot2 = np.flip(y_root, axis=0)
    y_root = yroot1 * y_root_l + yroot2 * (1. - y_root_l)

    # y_vowel
    yvowel1 = y_vowel
    yvowel2 = np.flip(y_vowel, axis=0)
    y_vowel = yvowel1 * y_vowel_l + yvowel2 * (1. - y_vowel_l)

    # y_consonant
    yconsonant1 = y_consonant
    yconsonant2 = np.flip(y_consonant, axis=0)
    y_consonant = yconsonant1 * y_consonant_l + yconsonant2 * (1. - y_consonant_l)

    return x, y_root, y_vowel, y_consonant

class TrainDataGenerator(keras.utils.Sequence):
    def __init__(self, X_set, Y_set, ids, batch_size = 16, img_size = (512, 512, 3), img_dir = TRAIN_DIR, augment = False, *args, **kwargs):
        self.X = X_set
        self.ids = ids
        self.Y = Y_set
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.augment = augment
        self.on_epoch_end()

        # Split Data
        self.x_indexed = self.X[self.ids]
        self.y_indexed = self.Y.iloc[self.ids]

        # Prep Y per Label   
        self.y_root = self.y_indexed.iloc[:,0:types['grapheme_root']].values
        self.y_vowel = self.y_indexed.iloc[:,types['grapheme_root']:types['grapheme_root']+types['vowel_diacritic']].values
        self.y_consonant = self.y_indexed.iloc[:,types['grapheme_root']+types['vowel_diacritic']:].values
    
    def __len__(self):
        return int(floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X, Y_root, Y_vowel, Y_consonant = self.__data_generation(indices)
        return X, {'root': Y_root, 'vowel': Y_vowel, 'consonant': Y_consonant}

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
    
    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size))
        Y_root = np.empty((self.batch_size, 168), dtype = np.int16)
        Y_vowel = np.empty((self.batch_size, 11), dtype = np.int16)
        Y_consonant = np.empty((self.batch_size, 7), dtype = np.int16)

        # Get Images for Batch
        for i, index in enumerate(indices):
            ID = self.x_indexed[index]
            image = _read(self.img_dir+ID+".png")
            
            X[i,] = image
        
        # Get Labels for Batch
        Y_root = self.y_root[indices]
        Y_vowel = self.y_vowel[indices]
        Y_consonant = self.y_consonant[indices]    

        if self.augment:
            X, Y_root, Y_vowel, Y_consonant = mixup(0.25, X, Y_root, Y_vowel, Y_consonant)

        return X, Y_root, Y_vowel, Y_consonant 

# Create Model
model = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

# Compile Model
model.compile(optimizer = Adam(lr = 0.0002),
                loss = {'root':                 'categorical_crossentropy',
                        'vowel':                'categorical_crossentropy',
                        'consonant':            'categorical_crossentropy'},
                loss_weights = {'root':         0.50,        
                                'vowel':        0.25,
                                'consonant':    0.25},
                metrics = {'root':              ['accuracy', tf.keras.metrics.Recall()],
                            'vowel':            ['accuracy', tf.keras.metrics.Recall()],
                            'consonant':        ['accuracy', tf.keras.metrics.Recall()] })

# Model Summary
print(model.summary())

# Multi Label Stratified Split stuff...
msss = MultilabelStratifiedShuffleSplit(n_splits = EPOCHS, test_size = TEST_SIZE, random_state = SEED)

# CustomReduceLRonPlateau function
best_val_loss = np.Inf
def CustomReduceLRonPlateau(model, history, epoch):
    global best_val_loss
    
    # ReduceLR Constants
    monitor = 'val_root_loss'
    patience = 12
    factor = 0.70
    min_lr = 1e-5

    # Get Current LR
    current_lr = float(K.get_value(model.optimizer.lr))
    
    # Print Current Learning Rate
    print(f'Current LR: {current_lr}')

    # Monitor Best Value
    current_val_loss = history[monitor][-1]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
    print(f'Best Vall Loss: {best_val_loss}')

    # Track last values
    if len(history[monitor]) >= patience:
        last5 = history[monitor][-5:]
        print(f'Last: {last5}')
        best_in_last = min(last5)
        print(f'Min value in Last: {best_in_last}')

        # Determine correction
        if best_val_loss < best_in_last:
            best_val_loss = np.Inf
            new_lr = current_lr * factor
            if new_lr < min_lr:
                new_lr = min_lr
            print(f'ReduceLRonPlateau setting learning rate to: {new_lr}')
            K.set_value(model.optimizer.lr, new_lr)

# History Placeholder
history = {}

# Epoch Training Loop
for epoch, msss_splits in zip(range(0, EPOCHS), msss.split(X_train, Y_train)):
    print(f'=========== EPOCH {epoch}')

    # Get train and test index, shuffle train indexes.
    train_idx = msss_splits[0]
    valid_idx = msss_splits[1]
    np.random.shuffle(train_idx)
    print(f'Train Length: {len(train_idx)}   First 10 indices: {train_idx[:10]}')    
    print(f'Valid Length: {len(valid_idx)}    First 10 indices: {valid_idx[:10]}')

    # Create Data Generators for Train and Valid
    data_generator_train = TrainDataGenerator(X_train, 
                                            Y_train,
                                            train_idx, 
                                            BATCH_SIZE, 
                                            (HEIGHT_NEW, WIDTH_NEW, CHANNELS),
                                            img_dir = TRAIN_DIR,
                                            augment = True)
    data_generator_val = TrainDataGenerator(X_train, 
                                            Y_train,
                                            valid_idx,
                                            2 * BATCH_SIZE, 
                                            (HEIGHT_NEW, WIDTH_NEW, CHANNELS),
                                            img_dir = TRAIN_DIR,
                                            augment = False)

    TRAIN_STEPS = int(len(data_generator_train))
    VALID_STEPS = int(len(data_generator_val))
    print(f'Train Generator Size: {len(data_generator_train)}')
    print(f'Validation Generator Size: {len(data_generator_val)}')
    
    model.fit_generator(generator = data_generator_train,
                        validation_data = data_generator_val,
                        steps_per_epoch = int(TRAIN_STEPS / 2.),
                        validation_steps = VALID_STEPS,
                        epochs = 1,
                        callbacks = [ModelCheckpointFull(RUN_NAME + 'model_' + str(epoch) + '.h5')],
                        verbose = 1)

    # Set and Concat Training History
    temp_history = model.history.history
    if epoch == 0:
        history = temp_history
    else:
        for k in temp_history: history[k] = history[k] + temp_history[k]

    # Custom ReduceLRonPlateau
    CustomReduceLRonPlateau(model, history, epoch)

    # Cleanup
    del data_generator_train, data_generator_val, train_idx, valid_idx
    gc.collect()

# Plot Training Summaries
plot_summaries(history, PLOT_NAME1, PLOT_NAME2)

# Create Predictions based on single model.
row_ids, targets = [], []
id = 0

# Loop through parquet files
for i in range(4):
    img_df = pd.read_parquet(os.path.join(DATA_DIR, 'test_image_data_' + str(i) + '.parquet'))
    img_df = img_df.drop('image_id', axis = 1)
    
    # Loop through rows in parquet file
    for index, row in img_df.iterrows():
        img = resize_image(row.values, WIDTH, HEIGHT, WIDTH_NEW, HEIGHT_NEW)
        img = np.stack((img,)*CHANNELS, axis=-1)
        image = img.reshape(-1, HEIGHT_NEW, WIDTH_NEW, 3)
        
        # Predict
        preds = model.predict(image, verbose = 1)
        for k in range(3):
            row_ids.append('Test_' + str(id) + '_' + tgt_cols[k])
            targets.append(np.argmax(preds[k]))
        id += 1

# Create and Save Submission File
submission = pd.DataFrame({'row_id': row_ids, 'target': targets}, columns = ['row_id', 'target'])
submission.to_csv('submission.csv', index = False)
print(submission.head(50))