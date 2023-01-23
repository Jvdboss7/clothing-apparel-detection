import os
import torch
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'clothing-apparel'
SAVED_MODEL_DIR = "PredictModel"
ZIP_FILE_NAME = 'clothing-dataset.zip'
DATA_DIR = "data"
ANNOTATIONS_COCO_JSON_FILE = '_annotations.coco.json'

INPUT_SIZE = 416
HORIZONTAL_FLIP = 0.3
VERTICAL_FLIP = 0.3
RANDOM_BRIGHTNESS_CONTRAST = 0.1
COLOR_JITTER = 0.1
BBOX_FORMAT = 'coco'

RAW_FILE_NAME = 'clothing'

# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'train'
DATA_INGESTION_TEST_DIR = 'test'
DATA_INGESTION_VALID_DIR = 'valid'

# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_DIR = 'Train'
DATA_TRANSFORMATION_TEST_DIR = 'Test'
DATA_TRANSFORMATION_TRAIN_FILE_NAME = "train.pkl"
DATA_TRANSFORMATION_TEST_FILE_NAME = "test.pkl"
DATA_TRANSFORMATION_TRAIN_SPLIT = 'train'
DATA_TRANSFORMATION_TEST_SPLIT = 'test'

# Model Training Constants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'model.pt'
TRAINED_BATCH_SIZE = 1
TRAINED_SHUFFLE = False
TRAINED_NUM_WORKERS = 2
EPOCH = 20

# Model evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
MODEL_EVALUATION_FILE_NAME = 'loss.csv'
BEST_MODEL_DIR = "best_model"
BUCKET_FOLDER_NAME="ModelTrainerArtifacts/trained_model/"

# Common constants
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Prediction Constants
#PREDICTION_CLASSES = [ 'clothing-prediction', 'bag', 'dress', 'hat', 'jacket', 'pants', 'shirt', 'shoe', 'shorts', 'skirt', 'sunglass']
PREDICTION_CLASSES = [ 'bag', 'dress', 'hat', 'jacket', 'pants', 'shirt', 'shoe', 'shorts', 'skirt', 'sunglass']

# AWS CONSTANTS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"