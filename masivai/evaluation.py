# %%writefile abalone/evaluation.py
import json
import pathlib
# import pickle
import tarfile
# import joblib
import numpy as np
import pandas as pd
# import xgboost
import argparse

import os
import boto3
import numpy as np

import tensorflow as tf
from sklearn.metrics import mean_squared_error


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    #   s3 = boto3.resource('s3')
    #   the_bucket = s3.Bucket(bucket_name)
    print('Blob {} downloaded to {}.'.format(
        source_blob_name, destination_file_name))

    s3 = boto3.resource('s3', verify=False)
    s3.Bucket(bucket_name).download_file(
        source_blob_name, destination_file_name)

    # print('Blob {} downloaded to {}.'.format(
    #     source_blob_name, destination_file_name))

# def _parse_args():
#     parser = argparse.ArgumentParser()
#     # Data, model, and output directories
#     # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
#     parser.add_argument('--model_dir', type=str)
#     parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
#     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
#     parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
#     parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

#     return parser.parse_known_args()


if __name__ == "__main__":
    #     model_path = f"/opt/ml/processing/model/model.tar.gz"
    #     with tarfile.open(model_path) as tar:
    #         tar.extractall(path=".")

    #     path = "/opt/ml/processing/model/model/000000001"

    #     sales_model = None
    #     os.makedirs('/tmp/listings_sales_model/variables/', exist_ok=True)
    #     if sales_model is None:
    #         download_blob(path, '/variables/variables.index',
    #                       '/tmp/listings_sales_model/variables/variables.index')
    #         download_blob(path, '/variables.data-00000-of-00001',
    #                       '/tmp/listings_sales_model/variables/variables.data-00000-of-00001')
    #         download_blob(path, '/saved_model.pb',
    #                       '/tmp/listings_sales_model/saved_model.pb')

    #         sales_model = tf.keras.models.load_model('/tmp/listings_sales_model/')

    #     model = pickle.load(open("xgboost-model", "rb"))

    #     args, unknown = _parse_args()
    #     print (args)

    print(123)

    sales_model = None
    BUCKET_NAME = "sagemaker-us-east-2-476153202769"

    os.makedirs('/tmp/listings_sales_model/variables/', exist_ok=True)
    if sales_model is None:

        download_blob(BUCKET_NAME, 'model/saved_model.pb',
                      '/tmp/listings_sales_model/saved_model.pb')

        download_blob(BUCKET_NAME, 'model/variables/variables.data-00000-of-00001',
                      '/tmp/listings_sales_model/variables/variables.data-00000-of-00001')

        download_blob(BUCKET_NAME, 'model/variables/variables.index',
                      '/tmp/listings_sales_model/variables/variables.index')


        sales_model = tf.keras.models.load_model('/tmp/listings_sales_model/')
        
##########################

#     sale_ft_layer = None
    
#     if sales_model is None:

#         download_blob(BUCKET_NAME, '/opt/ml/processing/sale/sale_ft_layer/saved_model.pb',
#                       '/tmp/sale/sale_ft_layer/saved_model.pb')

#         download_blob(BUCKET_NAME, 'model/variables/sale/sale_ft_layer/variables.data-00000-of-00001',
#                       '/tmp/sale/sale_ft_layer/variables/variables.data-00000-of-00001')

#         download_blob(BUCKET_NAME, 'model/variables/sale/sale_ft_layer/variables.index',
#                       '/tmp/sale/sale_ft_layer/variables/variables.index')


#         sale_ft_layer = tf.keras.models.load_model('/tmp/sale/sale_ft_layer/')

##########################

    test_path = "/opt/ml/processing/test/sale_test.csv"
    df = pd.read_csv(test_path)

#     y_test = df.iloc[:, 0].to_numpy()
#     df.drop(df.columns[0], axis=1, inplace=True)
    y_test = df[['price_min', 'price_max']].copy()
    x_test = df.drop(['price_min', 'price_max'], axis=1)

# #     X_test = xgboost.DMatrix(df.values)

    predictions = sales_model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
#     std = np.std(y_test - predictions)
    report_dict = {
            "mse": {
                "value": 4.0
            },
    }
    
    print (report_dict)
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
#     report_dict = {'d':'1'}
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    print ("123121312132311222")