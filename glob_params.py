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
