import os
import io
import sys
from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from clothing.exception import CustomException
from clothing.logger import logging
from clothing.constants import *
from clothing.configuration.s3_syncer import S3Sync
from clothing.entity.config_entity import ModelEvaluationConfig

class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.model_evaluation_config =ModelEvaluationConfig()

    def image_loader(self, image_bytes):
        """load image, returns cuda tensor"""
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            # image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = Image.open(io.BytesIO(image_bytes))
            convert_tensor = transforms.ToTensor()
            tensor_image = convert_tensor(image)
            # image = image[:3]
            image_int = torch.tensor(tensor_image * 255, dtype=torch.uint8)
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return tensor_image, image_int

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model_from_s3(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket 
            prediction_model_path=os.path.join("PredictModel")
            os.makedirs(prediction_model_path, exist_ok=True)
            s3_sync = S3Sync()

            s3_sync.sync_folder_from_s3(folder=prediction_model_path,bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,bucket_folder_name=self.model_evaluation_config.BUCKET_FOLDER_NAME)

            best_model_path = os.path.join(prediction_model_path, "model.pt")

            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def prediction(self, best_model_path: str, image_tensor, image_int_tensor) -> float:
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            model = torch.load(best_model_path, map_location=torch.device(DEVICE))
            model.eval()
            with torch.no_grad():
                prediction = model([image_tensor.to(DEVICE)])
                pred = prediction[0]

            bbox_tensor = draw_bounding_boxes(image_int_tensor,
                                pred['boxes'][pred['scores'] > 0.4],
                                [PREDICTION_CLASSES[i] for i in pred['labels'][pred['scores'] > 0.4].tolist()],
                                width=4).permute(0, 2, 1)

            transform = transforms.ToPILImage()
            img = transform(bbox_tensor)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())

            logging.info("Exited the prediction method of PredictionPipeline class")
            return img_str

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image, image_int = self.image_loader(data)
            print(image.shape)
            print(image_int.shape)
            best_model_path: str = self.get_model_from_s3()
            detected_image = self.prediction(best_model_path, image, image_int)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image
        except Exception as e:
            raise CustomException(e, sys) from e

