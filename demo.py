from clothing.entity.config_entity import DataIngestionConfig
from clothing.configuration.s3_operations import S3Operation
from clothing.components.data_ingestion import DataIngestion

data = DataIngestion(data_ingestion_config = DataIngestionConfig(), s3_operations = S3Operation())
data.initiate_data_ingestion()