# from clothing.pipeline.train_pipeline import TrainPipeline


# if __name__ == "__main__":
#     TrainPipeline().run_pipeline()

from clothing.configuration import s3_syncer


if __name__ == "__main__":
    s3_syncer.S3Sync().s3_key_path_available("clothing-apparel","ModelTrainerArtifacts/trained_model/model.pt")