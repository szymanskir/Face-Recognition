# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


def create_knn_model(neighbors_count, train_features, train_labels):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors_count)
    knn_model.fit(train_features, train_labels)
    return(knn_model)


@click.command()
@click.argument('train_faces_filepath', type=click.Path(exists=True))
@click.argument('train_labels_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())
@click.argument('neighbors_count', type=click.INT)
def main(train_faces_filepath,
         train_labels_filepath,
         output_model_filepath,
         neighbors_count):
    """
    Trains a knn classifier.
    """
    logger = logging.getLogger(__name__)
    train_faces = pd.read_csv(train_faces_filepath)
    train_labels= pd.read_csv(train_labels_filepath)
    logger.info('Training KNN model...')
    knn_model = create_knn_model(neighbors_count,
                                 train_faces,
                                 train_labels)

    logger.info(f'Saving KNN model to {output_model_filepath}...')
    joblib.dump(knn_model, output_model_filepath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
