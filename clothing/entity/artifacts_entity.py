from dataclasses import dataclass
# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str
    valid_file_path: str

# Data Transformation artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str 
    transformed_test_object: str
    number_of_classes: int

@dataclass
class ModelTrainerConfig:
     def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)
        self.BATCH_SIZE: int = TRAINED_BATCH_SIZE
        self.SHUFFLE: bool = TRAINED_SHUFFLE
        self.NUM_WORKERS = TRAINED_NUM_WORKERS
        self.EPOCH: int = EPOCH
        self.DEVICE = DEVICE 
