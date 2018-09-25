# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from sklearn.externals import joblib


def predict(model, test_features):
    labels_predicted = model.predict(test_features)
    return(labels_predicted)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('test_faces_filepath', type=click.Path(exists=True))
@click.argument('prediction_filepath', type=click.Path())
def main(model_filepath,
         test_faces_filepath,
         prediction_filepath):
    """
    Calculates predictions using the input model.
    """
    logger = logging.getLogger(__name__)
    model = joblib.load(model_filepath)
    test_faces = pd.read_csv(test_faces_filepath)

    logger.info(f'Predicting labels using {os.path.basename(model_filepath)}...')
    prediction = pd.DataFrame(predict(model, test_faces))
    logger.info(f'Saving results to {prediction_filepath}...')
    prediction.to_csv(prediction_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
