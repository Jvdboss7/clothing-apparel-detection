
import sys
from clothing.exception import CustomException
from clothing.logger import logging
from clothing.entity.config_entity import ModelPusherConfig
from clothing.entity.artifacts_entity import ModelPusherArtifacts
from clothing.configuration.s3_syncer import S3Sync


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig):

        self.model_pusher_config = model_pusher_config

        self.s3 = S3Sync()

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher.

            Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            self.s3.sync_folder_to_s3(folder=self.model_pusher_config.TRAINED_MODEL_DIR,bucket_name=self.model_pusher_config.BUCKET_NAME,bucket_folder_name=self.model_pusher_config.S3_MODEL_KEY_PATH)

            logging.info("Uploaded best model to s3 bucket")


            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

