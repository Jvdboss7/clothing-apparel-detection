import os
import logging
from clothing.exception import CustomException
from mypy_boto3_s3.service_resource import Bucket
from clothing.configuration.aws_connection import S3Client
import sys
class S3Sync:

    def __init__(self):
        self.s3_client = S3Client()
    
        self.s3_resource = self.s3_client.s3_resource

        self.s3_client = self.s3_client.s3_client


    def sync_folder_to_s3(self,folder,bucket_name,bucket_folder_name):
        command = f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}"

        logging.info(f"Command is : {command}")

        os.system(command)

    def sync_folder_from_s3(self,folder,bucket_name,bucket_folder_name):
        command = f"aws s3 sync s3://{bucket_name}/{bucket_folder_name} {folder}"

        logging.info(f"Command is : {command}")

        os.system(command)

        return folder

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)

            file_objects = [
                file_object for file_object in bucket.objects.filter(Prefix=s3_key)
            ]

            logging.info(f"{file_objects}")

            if len(file_objects) > 0:
                return True

            else:
                return False

        except Exception as e:
            raise CustomException(e, sys)

    def get_bucket(self, bucket_name: str):
        """
        Method Name :   get_bucket
        Description :   This method gets the bucket object based on the bucket_name
        Output      :   Bucket object is returned based on the bucket name
        On Failure  :   Write an exception log and then raise an exception
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered the get_bucket method of SimpleStorageService class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)

            logging.info("Exited the get_bucket method of SimpleStorageService class")

            return bucket

        except Exception as e:
            raise CustomException(e, sys) from e