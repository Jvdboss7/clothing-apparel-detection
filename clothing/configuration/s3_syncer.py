import os
import logging
class S3Sync:


    def sync_folder_to_s3(self,folder,bucket_name,bucket_folder_name):
        command = f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}"

        logging.info(f"Command is : {command}")

        os.system(command)

    def sync_folder_from_s3(self,folder,bucket_name,bucket_folder_name):
        command = f"aws s3 sync s3://{bucket_name}/{bucket_folder_name} {folder}"

        logging.info(f"Command is : {command}")

        os.system(command)
