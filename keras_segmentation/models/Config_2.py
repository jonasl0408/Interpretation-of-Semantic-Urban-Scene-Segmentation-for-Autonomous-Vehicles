'''
TODO:
    Configurations for Attention gated residual UNet
    on spatial-level or channel-level
    Hyper parameters
'''

__author__ = 'MoleImg'

# Main thread controllers
MODEL_NAME = 'AttentionSEResUNet'   # AttentionResUNet/AttentionSEResUNet
TRAIN_FLAG = False # if training
TEST_FLAG = True # if test
MODEL_SAVE_FLAG = False # if saving model
MODEL_LOAD_FLAG = False # if load training checkpoint
TRAINING_VISUAL_FLAG = False # if activate training visualization
TENSORBOARD_FLAG = True # if use Tensorboard

# input data
INPUT_SIZE = 256
INPUT_CHANNEL = 3   # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = 13

# network structure
FILTER_NUM = 32 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
DOWN_SAMP_SIZE = 2 # size of pooling filters
UP_SAMP_SIZE = 2 # size of upsampling filters

# network hyper-parameter
DROPOUT_RATE = 0
BATCH_NORM_FLAG = True

# training data
TRAINING_SIZE = 2975
TRAINING_START = 1
BATCH_SIZE = 50
EPOCH = 10
VALIDATION_SPLIT = 0.10

# test data
TEST_SIZE = 1
TEST_START = 1
