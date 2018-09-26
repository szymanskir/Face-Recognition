# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from sklearn.decomposition import PCA


def create_pca_model(number_of_components, train_faces):
    pca = PCA(n_components=number_of_components, random_state=0)
    pca.fit(train_faces)

    return(pca)


@click.command()
@click.argument('input_train_faces', type=click.Path(exists=True))
@click.argument('input_test_faces', type=click.Path(exists=True))
@click.argument('output_train_faces', type=click.Path())
@click.argument('output_test_faces', type=click.Path())
@click.argument('number_of_components', type=click.INT)
def main(input_train_faces,
         input_test_faces,
         output_train_faces,
         output_test_faces,
         number_of_components):
    """
    Extracts features using the PCA decomposition algorithm (eigenfaces)
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Extracting features using the PCA algorithm(n={number_of_components})...')

    train_faces = pd.read_csv(input_train_faces)
    test_faces = pd.read_csv(input_test_faces)

    pca = create_pca_model(number_of_components, train_faces)

    train_features = pd.DataFrame(pca.transform(train_faces))
    test_features = pd.DataFrame(pca.transform(test_faces))

    logging.info("Finished extracting features")
    logging.info("Saving extracted features")

    train_features.to_csv(output_train_faces, index=False)
    test_features.to_csv(output_test_faces, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
