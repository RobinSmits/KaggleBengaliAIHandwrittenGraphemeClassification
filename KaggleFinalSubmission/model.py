import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model, Input
from keras.layers import Dense, Lambda, BatchNormalization, Dropout 
import efficientnet.keras as efn

# Generalized mean pool - GeM
gm_exp = tf.Variable(3.0, dtype = tf.float32)
def generalized_mean_pool_2d(X):
    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 
                        axis = [1, 2], 
                        keepdims = False) + 1.e-7)**(1./gm_exp)
    return pool

def create_model(input_shape):
    # Input Layer
    input = Input(shape = input_shape)
    
    # Create and Compile Model and show Summary
    x_model = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = input, pooling = None, classes = None)
    
    # Root
    lambda_layer1 = Lambda(generalized_mean_pool_2d)
    lambda_layer1.trainable_weights.extend([gm_exp])
    x1 = lambda_layer1(x_model.output)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(512, activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    
    # Vowel
    lambda_layer2 = Lambda(generalized_mean_pool_2d)
    lambda_layer2.trainable_weights.extend([gm_exp])
    x2 = lambda_layer2(x_model.output)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(512, activation = 'relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    
    # Consonant
    lambda_layer3 = Lambda(generalized_mean_pool_2d)
    lambda_layer3.trainable_weights.extend([gm_exp])
    x3 = lambda_layer3(x_model.output)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)
    x3 = Dense(512, activation = 'relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)

    # multi output
    grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x1)
    vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x2)
    consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x3)

    # model
    model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])

    return model