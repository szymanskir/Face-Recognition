# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd

from glob import glob
from sklearn.metrics import accuracy_score


@click.command()
@click.argument('predictions_filepath', type=click.Path(exists=True))
@click.argument('test_labels_filepath', type=click.Path(exists=True))
@click.argument('summary_filepath', type=click.Path())
def main(predictions_filepath,
         test_labels_filepath,
         summary_filepath):
    """
    Summarizes prediction results.
    """
    logger = logging.getLogger(__name__)
    prediction_list = glob(os.path.join(predictions_filepath, '*.csv'))
    test_labels = pd.read_csv(test_labels_filepath)
    summary = dict()

    for prediction_filepath in prediction_list:
        logger.info(f'Evaluating accuracy for {prediction_filepath}...')
        prediction = pd.read_csv(prediction_filepath)
        accuracy = accuracy_score(test_labels, prediction)
        summary[prediction_filepath] = accuracy

    summary = pd.DataFrame({'Filename': list(summary.keys()),
                            'Score': list(summary.values())})
    logger.info(f'Saving summary to {summary_filepath}..')
    summary.sort_values(by='Score', ascending=False).to_csv(summary_filepath, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
