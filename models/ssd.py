"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate, Reshape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Softmax


def build_base_network(input_shape=(None,None,3), num_units=16):
    inputs = Input(shape=input_shape)

    # BLOCK 1
    conv1_1 = Conv2D(num_units, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', name='conv1_1')(inputs)
    norm1_1 = BatchNormalization(name='norm1_1')(conv1_1)
    conv1_2 = Conv2D(num_units, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', name='conv1_2')(norm1_1)
    norm1_2 = BatchNormalization(name='norm1_2')(conv1_2)
    conv1_3 = Conv2D(num_units, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', name='conv1_3')(norm1_2)
    norm1_3 = BatchNormalization(name='norm1_3')(conv1_3)

    # BLOCK 2
    conv2_1 = Conv2D(num_units * 2, (3, 3), activation='relu',
                     padding='same', name='conv2_1')(norm1_3)
    norm2_1 = BatchNormalization(name='norm2_1')(conv2_1)
    conv2_2 = Conv2D(num_units * 2, (3, 3), strides=(2, 2), activation='relu',
                     padding='same', name='conv2_2')(norm2_1)
    norm2_2 = BatchNormalization(name='norm2_2')(conv2_2)

    # BLOCK 3
    conv3_1 = Conv2D(num_units * 4, (3, 3), activation='relu',
                     padding='same', name='conv3_1')(norm2_2)
    norm3_1 = BatchNormalization(name='norm3_1')(conv3_1)
    conv3_2 = Conv2D(num_units * 4, (3, 3), strides=(2, 2), activation='relu',
                     padding='same', name='conv3_2')(norm3_1)
    norm3_2 = BatchNormalization(name='norm3_2')(conv3_2)

    # BLOCK 4
    conv4_1 = Conv2D(num_units * 8, (3, 3), activation='relu',
                     padding='same', name='conv4_1')(norm3_2)
    norm4_1 = BatchNormalization(name='norm4_1')(conv4_1)
    conv4_2 = Conv2D(num_units * 8, (3, 3), strides=(2, 2), activation='relu',
                     padding='same', name='conv4_2')(norm4_1)
    norm4_2 = BatchNormalization(name='norm4_2')(conv4_2)

    # Block 5
    conv5_1 = Conv2D(num_units * 8, (3, 3), activation='relu',
                     padding='same', name='conv5_1')(norm4_2)
    norm5_1 = BatchNormalization(name='norm5_1')(conv5_1)
    conv5_2 = Conv2D(num_units * 8, (3, 3), strides=(2, 2), activation='relu',
                     padding='same', name='conv5_2')(norm5_1)
    norm5_2 = BatchNormalization(name='norm5_2')(conv5_2)

    outputs = norm5_2

    return Model(inputs, outputs, name='base_network')


def attach_multibox_head(base_network, source_layer_names,
                         num_priors=4, num_classes=10):
    heads = []
    for idx, layer_name in enumerate(source_layer_names):
        source_layer = base_network.get_layer(layer_name).output

        # Classification
        clf = Conv2D(num_priors * (num_classes+1), (3, 3),
                     padding='same', name=f'clf_head{idx}_logit')(source_layer)
        clf = Reshape((-1, num_classes+1),
                      name=f'clf_head{idx}_reshape')(clf)
        clf = Softmax(axis=-1, name=f'clf_head{idx}')(clf)

        # Localization
        loc = Conv2D(num_priors * 4, (3,3), padding='same',
                     name=f'loc_head{idx}')(source_layer)
        loc = Reshape((-1,4),
                      name=f'loc_head{idx}_reshape')(loc)
        head = Concatenate(axis=-1, name=f'head{idx}')([clf, loc])
        heads.append(head)
    if len(heads) > 1:
        predictions = Concatenate(axis=1, name='predictions')(heads)
    else:
        predictions = K.identity(heads[0],name='predictions')
    return predictions