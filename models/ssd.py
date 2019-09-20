"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
from tensorflow.python.keras.layers import Input, Layer
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate, Reshape
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Softmax, Add
from tensorflow.python.keras.activations import sigmoid


def build_base_network(input_shape=(128,128,3), num_units=16): # None,None,3
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


def remodel_fpn_network(base_network, source_layer_names, num_features=64):
    upsampled = None

    pyramid_outputs = []
    num_layer = len(source_layer_names)
    for idx, layer_name in enumerate(source_layer_names[::-1]):
        source_layer = base_network.get_layer(layer_name).output

        lateral = Conv2D(num_features, (1, 1), padding='same')(source_layer)
        if upsampled is not None:
            output = Add()([lateral, upsampled])
            upsampled = UpSampling2D((2, 2))(output)
        else:
            output = lateral
            upsampled = UpSampling2D((2, 2))(lateral)
        p_num = num_layer - idx
        pyramid_output = Conv2D(num_features, (3, 3),
                                padding='same', name=f'P{p_num}')(output)
        pyramid_outputs.append(pyramid_output)

    fpn_network = Model(base_network.input, pyramid_outputs)
    return fpn_network


def attach_multibox_head(network, source_layer_names,
                         num_priors=4, num_classes=11, activation='softmax'): # 10->11
    heads = []
    # batch_size = 64 # OpenCV에서 인식을못해서 하드코딩한다 reshape를 인식못한다
    for idx, layer_name in enumerate(source_layer_names):
        source_layer = network.get_layer(layer_name).output

        # OpenCV Loading Error
        # "Can't create layer \"loc_head2_reshape_2/Shape\" of type \"Shape\""
        # 조치 : class개수를 10에서 11로 바꿔줌

        w = source_layer.get_shape().as_list()[1]
        h = source_layer.get_shape().as_list()[2]
        print("w : ", w)
        print("h : ", h)
        print("num_priors : ", num_priors)
        # Classification
        clf = Conv2D(num_priors * num_classes, (3, 3),
                     padding='same', name=f'clf_head{idx}_logit')(source_layer)
        print("clf shape입니다 : ", clf.shape)
        clf = Reshape((w*h*num_priors, num_classes), name=f'clf_head{idx}_reshape')(clf)  # (-1, num_classes) # w*h*num_priors
        print("clf의 reshape 후입니다 : ", clf.shape)
        if activation == 'softmax':
            clf = Softmax(axis=-1, name=f'clf_head{idx}')(clf)
        elif activation == 'sigmoid':
            clf = sigmoid(clf)
        else:
            raise ValueError('activation은 {softmax,sigmoid}중에서 되어야 합니다.')

        # Localization
        loc = Conv2D(num_priors * 4, (3,3), padding='same',
                     name=f'loc_head{idx}')(source_layer)
        print("loc의 shape입니다 : ", loc.shape)
        loc = Reshape((w*h*num_priors, 4), name=f'loc_head{idx}_reshape')(loc)  #Reshape((-1, 4),
        print("loc의 reshape 후입니다 : ", loc.shape)
        head = Concatenate(axis=-1, name=f'head{idx}')([clf, loc])
        heads.append(head)

    if len(heads) > 1:
        predictions = Concatenate(axis=1, name='predictions')(heads)
    else:
        predictions = K.identity(heads[0],name='predictions')
    return predictions

