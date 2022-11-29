import os
import sys
import math
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from clothing.logger import logging
from clothing.exception import CustomException
from clothing.utils.main_utils import load_object
from clothing.ml.models.model_optimiser import model_optimiser
from clothing.entity.config_entity import ModelTrainerConfig
from clothing.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from clothing.ml.detection.engine import train_one_epoch
from torchvision.models.detection import retinanet_resnet50_fpn_v2

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                    model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def train(self, model, optimizer, loader, device, epoch):
        try:
            model.to(device)
            model.train() 
            all_losses = []
            all_losses_dict = []
            
            for images, targets in tqdm(loader):
                images = list(image.to(device) for image in images)
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
                losses = sum(loss for loss in loss_dict.values())
                loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                loss_value = losses.item()
                
                all_losses.append(loss_value)
                all_losses_dict.append(loss_dict_append)
                
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")  # train if loss becomes infinity
                    print(loss_dict)
                    sys.exit(1)
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def collate_fn(batch):
        """
        This is our collating function for the train dataloader, 
        it allows us to create batches of data that can be easily pass into the model
        """
        try:
            return tuple(zip(*batch))
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            train_dataset = load_object(self.data_transformation_artifacts.transformed_train_object)

            train_loader = DataLoader(train_dataset,
                                     batch_size=self.model_trainer_config.BATCH_SIZE,
                                     shuffle=self.model_trainer_config.SHUFFLE,
                                     num_workers=self.model_trainer_config.NUM_WORKERS,
                                     collate_fn=self.collate_fn
                                     )

            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)

            test_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=self.model_trainer_config.SHUFFLE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS,
                                      collate_fn=self.collate_fn
                                      )

            logging.info("Loaded training data loader object")
            model = retinanet_resnet50_fpn_v2(score_thresh=0.5,num_classes=11)

            optimiser = model_optimiser(model)

            logging.info("loaded optimiser")

            for epoch in range(self.model_trainer_config.EPOCH):

                train_one_epoch(model, optimiser, test_loader, self.model_trainer_config.DEVICE, epoch, print_freq=10)

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)

            logging.info(f"Saved the trained model")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e



