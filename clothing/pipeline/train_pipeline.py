import sys
from clothing.components.data_ingestion import DataIngestion
from clothing.components.data_transformation import DataTransformation
from clothing.configuration.s3_operations import S3Operation
from clothing.exception import CustomException
from clothing.entity.config_entity import DataIngestionConfig, DataTransformationConfig
from clothing.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts
from clothing.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.s3_operations=S3Operation()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from S3 bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config, s3_operations= S3Operation()
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train, test and valid from s3")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifacts,) -> DataTransformationArtifacts:
        logging.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )
        try:
            data_transformation = DataTransformation(
                
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            
        except Exception as e:
            raise CustomException(e, sys) from e