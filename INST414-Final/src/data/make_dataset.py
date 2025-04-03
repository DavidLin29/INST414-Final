import click
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Processing raw data to create cleaned data')

    # Load the raw data
    df = pd.read_csv(input_filepath)

    # Clean the data
    logger.info('Cleaning the data...')
    df = clean_data(df)

    # Save the cleaned data
    df.to_csv(output_filepath, index=False)
    logger.info(f'Cleaned data saved to {output_filepath}')


def clean_data(df):
    """Clean the dataset by removing unnecessary columns, handling missing values, etc."""
    df.drop(columns=["AdvertisingPlatform", "AdvertisingTool"], inplace=True)

    # Handle missing values
    missing = df.isnull()
    missing_amount = missing.sum()
    print(missing_amount)

    # Feature engineering (same as before)
    df["Age Category"] = pd.cut(df["Age"], bins=[13, 23, 33, 43, 53, 63, 73],
                                labels=["13-23", "23-33", "33-43", "43-53", "53-63", "63-73"], right=False)
    df["EngagementScore"] = MinMaxScaler().fit_transform(df[["WebsiteVisits", "PagesPerVisit", "TimeOnSite", 
                                                           "EmailOpens", "EmailClicks", "SocialShares"]]).sum(axis=1)
    df["EmailEngagementRatio"] = df["EmailClicks"].div(df["EmailOpens"].replace(0, 0), fill_value=0)

    return df


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Load .env variables if necessary
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
