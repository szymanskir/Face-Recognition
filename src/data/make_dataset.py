# -*- coding: utf-8 -*-
import click
import logging
import os
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """
    Downloads the olivetti faces dataset and saves it in the output_filepath
    directory.
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading Olivetti faces...')
    olivetti_faces = fetch_olivetti_faces()

    data = pd.DataFrame(data=olivetti_faces.data)
    labels = pd.DataFrame(data=olivetti_faces.target)

    logger.info('Splitting dataset into training and testing sets...')
    train_data, train_labels, test_data, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=0)

    train_data.to_csv(os.path.join(output_filepath, 'face_data_train.csv'), index=False)
    train_labels.to_csv(os.path.join(output_filepath, 'labels_train.csv'), index=False)
    test_data.to_csv(os.path.join(output_filepath, 'face_data_test.csv'), index=False)
    test_labels.to_csv(os.path.join(output_filepath, 'labels_test.csv'), index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
