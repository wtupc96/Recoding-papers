import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ROOT = r'F:\recoding_papers'

# Dataset Paths.
DATASET_ROOT = os.path.join(ROOT, 'dataset')

MNIST_DATASET = os.path.join(DATASET_ROOT, 'mnist')
CIFAR10_DATASET = os.path.join(DATASET_ROOT, 'cifar-10-batches-py')
ENGLISH_HAND_DATASET = os.path.join(DATASET_ROOT, 'EnglishHnd', 'English', 'Hnd')


# Model Paths
MODEL_ROOT = os.path.join(ROOT, 'model')

INCEPTION_MDOEL = os.path.join(MODEL_ROOT, 'inception_v3_2016_08_28', 'inception_v3.ckpt')
MOBILE_MODEL = os.path.join(MODEL_ROOT, 'mobilenet_v2_1.4_224')
RESNET_MODEL = os.path.join(MODEL_ROOT, 'resnet_v2_152_2017_04_14', 'resnet_v2_152.ckpt')