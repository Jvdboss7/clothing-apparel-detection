import os

import boto3

from clothing.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    REGION_NAME,
)


class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):

        if S3Client.s3_resource == None or S3Client.s3_client == None:

            S3Client.s3_resource = boto3.resource(
                "s3",
            )

            S3Client.s3_client = boto3.client(
                "s3",
            )

        self.s3_resource = S3Client.s3_resource

        self.s3_client = S3Client.s3_client