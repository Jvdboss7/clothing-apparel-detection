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

# Model Trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str