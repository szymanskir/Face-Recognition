# -*- coding: utf-8 -*-
import click
import logging
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading Olivetti faces...')
    olivetti_faces = fetch_olivetti_faces(data_home=output_filepath)

    logger.info('Saving faces...')
    data = pd.DataFrame(data=olivetti_faces.data)
    data.to_csv(os.path.join(output_filepath, 'face_data.csv'))

    logger.info('Saving labels...')
    labels = pd.DataFrame(data=olivetti_faces.target)
    labels.to_csv(os.path.join(output_filepath, 'face_labels.csv'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
