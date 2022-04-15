from keras.models import *
from keras.layers import *
import tensorflow as tf

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Dropout
from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import add
from keras_segmentation.layers.attention import PAM, CAM


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1
    

def attention(tensor, att_tensor, n_filters=512):
    #g1 = conv2d(tensor, n_filters, kernel_size=kernel_size)
    g1 = Conv2D(n_filters, (1, 1), data_format=IMAGE_ORDERING, activation='relu', padding='valid')(tensor)
    #x1 = conv2d(att_tensor, n_filters, kernel_size=kernel_size)
    x1 = Conv2D(n_filters, (1, 1), strides=(2, 2), data_format=IMAGE_ORDERING, activation='relu', padding='valid')(att_tensor)
    net = tf.add(g1, x1)
    #net = tf.nn.relu(net)
    #net = conv2d(net, 1, kernel_size=kernel_size)
    net = Conv2D(1, (1, 1), data_format=IMAGE_ORDERING, activation='sigmoid', padding='valid')(net)
    #net = tf.nn.sigmoid(net)
    #net = tf.concat([att_tensor, net], axis=-1)
    net = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(net)
    net = net * att_tensor
    return net

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x


    
def Att_Unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    #o_ = (Conv2D(256, (3, 3), padding='same' , activation='relu' , data_format=IMAGE_ORDERING))(o)
        

    o_up = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = attention(o, f3, 256)
    print('o.shape', o.shape)
    print('o.shape', o.shape)
    o = (concatenate([o_up, o], axis=MERGE_AXIS))
    
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    #o_ = (Conv2D(128, (3, 3), padding='same' , activation='relu' , data_format=IMAGE_ORDERING))(o)


    o_up = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = attention(o, f2, 128)
    o = (concatenate([o_up, o], axis=MERGE_AXIS))
    
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    #o_ = (Conv2D(64, (3, 3), padding='same' , activation='relu' , data_format=IMAGE_ORDERING))(o)

    o_up = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = attention(o, f1, 64)

    if l1_skip_conn:
        o = (concatenate([o_up, o], axis=MERGE_AXIS))
    
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model


def DA_Net(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f3

    # ATTENTION
    pam = PAM()(o)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    pam = BatchNormalization(axis=3)(pam)
    pam = Activation('relu')(pam)
    pam = Dropout(0.5)(pam)
    pam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)

    cam = CAM()(o)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    cam = BatchNormalization(axis=3)(cam)
    cam = Activation('relu')(cam)
    cam = Dropout(0.5)(cam)
    cam = Conv2D(512, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)

    feature_sum = add([pam, cam])
    feature_sum = Dropout(0.5)(feature_sum)
    feature_sum = Conv2d_BN(feature_sum, 512, 1)
    #up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(feature_sum), 512, 2)
    merge7 = concatenate([f3, feature_sum], axis=3)
    conv7 = Conv2d_BN(merge7, 512, 3)
    conv7 = Conv2d_BN(conv7, 512, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
    merge8 = concatenate([f2, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 256, 3)
    conv8 = Conv2d_BN(conv8, 256, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
    merge9 = concatenate([f1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 64, 3)
    conv9 = Conv2d_BN(conv9, 64, 3)
    
    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 32, 2)
    conv10 = Conv2d_BN(up10, n_classes, 3)
    o = conv10
    #conv11 = Conv2d_BN(conv10, classes, 1, use_activation=None)
    #activation = Activation('sigmoid', name='Classification')(conv11)
    
    model = get_segmentation_model(img_input, o)

    return model

def unet_mini(n_classes, input_height=360, input_width=480, channels=3):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)

    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)

    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same' , name="seg_feats")(conv5)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv5)

    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_mini"
    return model


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))
    
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model


def unet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "unet"
    return model

def segnet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "unet"
    return model


def attention_unet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = Att_Unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "attention_unet"
    return model

def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_unet"
    return model

def attention_vggunet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = Att_Unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "attention_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_unet"
    return model

def da_resnet50_unet(n_classes, input_height=416, input_width=608, encoder_level=4, channels=3):

    model = DA_Net(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "da_resnet50_unet"
    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3, channels=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "mobilenet_unet"
    return model


if __name__ == '__main__':
    m = unet_mini(12)
    m = _unet(12, vanilla_encoder)
    # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _unet(12, get_vgg_encoder)
    m = _unet(12, get_resnet50_encoder)
