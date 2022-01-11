"""
This file contains property valuation models
for both sales and Rent, Both classes inherits the
common functionality from the base class
A separate get_data_labeled function is defined
to assign respective labels (sales/rent) before
training the model
"""

import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os
import json
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers


# class BaseModel:
    
#     def get_preprocessed_data(self, data, preprocessor_object):
        
#         return preprocessor_object.convert_dataframe_to_float_values(data)
    
    
#     def train_val_test_split(self, data, labels):
        
#         output_min = labels[0]
#         output_max = labels[1]
        
#         X_train, X_test, y_train, y_test = train_test_split(data, pd.DataFrame([output_min, output_max]).transpose(), test_size=0.2, random_state=42)
#         X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

#         return X_train, y_train, X_val, y_val, X_test, y_test
    
    
#     def model_definition(self, shape, learning_rate):
        
#         input_layer, output_layer = shape
        
#         self.model = tf.keras.models.Sequential([
#           tf.keras.layers.Flatten(input_shape=[input_layer]),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(20, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(12, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(10, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(7, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(5, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(3, activation='relu'),
#           tf.keras.layers.Dense(output_layer)
#         ])

#         opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#         self.model.compile(optimizer=opt,
#                       loss='mse',
#                       metrics=['MAE']
#                      )
    
#     def model_train(self, data, params):
        
#         X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]
#         batch_size, epochs = params
        
#         history = self.model.fit(x=X_train, y=y_train,
#                   validation_data=(X_val, y_val),
#                   batch_size=batch_size,
#                   epochs=epochs)
        
    
#     def evaluate(self, X_test, y_test):
        
#         return self.model.evaluate(X_test, y_test)
    
    
#     def predict(self, X_test):
        
#         return self.model.predict(X_test)
    
#     def model_save(self, model_name):
        
#         self.model.save(model_name + ".h5")
        
#     def model_load(self, model_name):
        
#         tf.keras.models.load_model(model_name + ".h5")
    
    
# class SalesModel(BaseModel):
    
#     def __init__(self, input_dataframe):
#         self.input_data = input_dataframe
    
#     def get_preprocessed_data(self, preprocessor_object):
        
#         all_data = preprocessor_object.remove_rows_with_no_labels(preprocessor_object.dataframe, "sale_max_m")
        
#         output_min = all_data["sale_mix_m"]
#         output_max = all_data["sale_max_m"]
    
#         data = copy.deepcopy(self.input_data)
#         data["sale_min_m"] = output_min
#         data["sale_max_m"] = output_max
#         return super(SalesModel, self).get_preprocessed_data(data, preprocessor_object)
    
    
#     def train_val_test_split(self, data):
        
#         output_max = data.pop("sale_max_m")
#         output_min = data.pop("sale_min_m")

#         X_train, y_train, X_val, y_val, X_test, y_test = super(SalesModel, self).train_val_test_split(data, [output_min, output_max])
        
#         return X_train, y_train, X_val, y_val, X_test, y_test
    
    
# class RentModel(BaseModel):
    
#     def __init__(self, input_dataframe):
#         self.input_data = input_dataframe
    
#     def get_preprocessed_data(self, preprocessor_object):

#         all_data = preprocessor_object.remove_rows_with_no_labels(preprocessor_object.dataframe, "sale_max_m")
        
#         output_min = all_data["rent_min_k"]
#         output_max = all_data["rent_max_k"]

        
#         output_min = preprocessor_object.convert_currency_values_in_lac_to_multiples_of_thousand(output_min)
#         output_max = preprocessor_object.convert_currency_values_in_lac_to_multiples_of_thousand(output_max)

#         data = copy.deepcopy(self.input_data)
#         data["rent_min_k"] = output_min
#         data["rent_max_k"] = output_max
        
#         return super(RentModel, self).get_preprocessed_data(data, preprocessor_object)
        
        
#     def train_val_test_split(self, data):
        
#         output_max = data.pop("rent_max_k")
#         output_min = data.pop("rent_min_k")
        
#         X_train, y_train, X_val, y_val, X_test, y_test = super(RentModel, self).train_val_test_split(data, [output_min, output_max])

#         return X_train, y_train, X_val, y_val, X_test, y_test
    
    
# class ListingsModel:
    
#     def __init__(self, input_dataframe):
#         self.input_data = input_dataframe    
    
#     def listings_train_test_split(self):
        
#         train, test = train_test_split(self.input_data, test_size=0.2)
#         return train, test
    
#     def df_to_dataset(self, dataframe, shuffle=True, batch_size=32):
        
#         dataframe = dataframe.copy()
#         try:
# #             labels = dataframe.pop('price_min')
#             labels = dataframe[['price_min', 'price_max']].copy()
#             dataframe = dataframe.drop(['price_min', 'price_max'], axis=1)
#         except:
#             labels = [[0,0]] * dataframe.shape[0]
# #             labels = [0] * dataframe.shape[0]
#         dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), dict(labels)))
        
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=len(dataframe))
#             dataset = dataset.batch(batch_size)
#         return dataset
    
#     def feature_columns(self):

#         numerical_column = ["general_size", "bed", "bath"]
# #         categorical_indicator_column = ["subsector", "purpose"]
#         categorical_indicator_column = ["subsector"]
#         categorical_embedding_column = ["sector", "type", "subtype"]

#         feature_columns = []

#         for header in numerical_column:
#             feature_columns.append(feature_column.numeric_column(header))

#         for feature_name in categorical_indicator_column:
#             vocabulary = self.input_data[feature_name].unique()
#             cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
#             one_hot = feature_column.indicator_column(cat_c)
#             feature_columns.append(one_hot)

#         for feature_name in categorical_embedding_column:
#             vocabulary = self.input_data[feature_name].unique()
#             cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
#             embeding = feature_column.embedding_column(cat_c, dimension=4)
#             feature_columns.append(embeding)

#         self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# #         return feature_layer
    
# #     def fill_null_data(self):
# #         self.input_data.fillna(0, inplace=True)
    
#     def model_definition(self, learning_rate):
        
#         self.model = tf.keras.models.Sequential([
# #           self.feature_layer,
# #           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Flatten(input_shape=[1,4]),
#           tf.keras.layers.Dense(12, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(10, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(8, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(7, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(5, activation='relu'),
#           tf.keras.layers.BatchNormalization(),
#           tf.keras.layers.Dense(3, activation='relu'),
#           tf.keras.layers.Dense(2)
#         ])

#         opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#         self.model.compile(optimizer=opt,
#                       loss='mse',
#                       metrics=['MAE']
#                      )

# #     def model_train(self, train_dataset, epochs):

# #         history = self.model.fit(train_dataset, epochs=epochs)

#     def model_train(self, X_train, y_train, X_val, y_val, epochs):
#         history = self.model.fit(x=X_train, y=y_train,
#           validation_data=(X_val, y_val),
#           batch_size=32,
#           epochs=1000)
        
        
#     def evaluate(self, data_dataset):
        
#         return self.model.evaluate(data_dataset)
    
    
#     def predict(self, data_dataset):
        
#         return self.model.predict(data_dataset)
    
#     def model_save(self, model_name):
        
#         self.model.save(model_name,save_format='tf')
        
#     def model_load(self, model_name):
        
#         self.model = tf.keras.models.load_model(model_name)
        
        
def listings_train_test_split(input_data):

    train, test = train_test_split(input_data, test_size=0.2)
    return train, test

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    
    dataframe = dataframe.copy()
    try:
#             labels = dataframe.pop('price_min')
        labels = dataframe[['price_min', 'price_max']].copy()
#         labels = dataframe[['price_min']].copy()
#         labels = dataframe["price_min"]
        dataframe = dataframe.drop(['price_min', 'price_max'], axis=1)
#         labels = [[0,0]] * dataframe.shape[0]        
    except:
        labels = [[0,0]] * dataframe.shape[0]
#             labels = [0] * dataframe.shape[0]
#     labels.rename(columns={'price_min': 'output_1', 'price_max': 'output_2'}, inplace=True)
#     labels.rename(columns={'price_min': 'output_1'}, inplace=True)
#     dataset = tf.data.Dataset.from_tensor_slices((dataframe, labels))
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    print (dataset)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
        dataset = dataset.batch(batch_size)
    return dataset

def feature_columns(input_data):

    numerical_column = ["general_size", "bed", "bath"]
#         categorical_indicator_column = ["subsector", "purpose"]
    categorical_indicator_column = ["subsector"]
    categorical_embedding_column = ["sector", "type", "subtype"]

    feature_columns = []

    for header in numerical_column:
        feature_columns.append(feature_column.numeric_column(header))

    for feature_name in categorical_indicator_column:
        vocabulary = input_data[feature_name].unique()
        cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
        one_hot = feature_column.indicator_column(cat_c)
        feature_columns.append(one_hot)

    for feature_name in categorical_embedding_column:
        vocabulary = input_data[feature_name].unique()
        cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
        embeding = feature_column.embedding_column(cat_c, dimension=4)
        feature_columns.append(embeding)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    return feature_layer
#     return feature_columns  

#     def fill_null_data(self):
#         self.input_data.fillna(0, inplace=True)

# def model_definition(feature_layer, X_train, y_train, learning_rate):
# #     print (feature_layer)
#     model = tf.keras.models.Sequential([
#       feature_layer,
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(12, activation='relu'),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(10, activation='relu'),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(8, activation='relu'),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(7, activation='relu'),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(5, activation='relu'),
#       tf.keras.layers.BatchNormalization(),
#       tf.keras.layers.Dense(3, activation='relu'),
#       tf.keras.layers.Dense(2)
#     ])

#     opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=opt,
#                   loss='mse',
#                   metrics=['MAE']
#                  )

#     model.fit(x=X_train, y=y_train, epochs=10)
    
#     m = tf.estimator.DNNRegressor(feature_columns=feature_columns, hidden_units=[12,10,8,7,5,3], label_dimension=2,
#     optimizer=lambda: tf.keras.optimizers.Adam(
#         learning_rate=tf.compat.v1.train.exponential_decay(
#             learning_rate=0.01,
#             global_step=tf.compat.v1.train.get_global_step(),
#             decay_steps=10000,
#             decay_rate=0.96)))
#     m.fit(x=X_train, y=y_train, epochs=10)
#     return m
    
        
#     return model


def model_definition(X_train, y_train, X_val, y_val, learning_rate):

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=[1,19]),
      tf.keras.layers.Dense(15, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(10, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(7, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(5, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(3, activation='relu'),
      tf.keras.layers.Dense(2)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=['MAE']
                 )
    print (X_train)
    print ("-------------------------------------------")
    print (y_train)
    model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)
        
    return model


#     def model_train(self, train_dataset, epochs):

#         history = self.model.fit(train_dataset, epochs=epochs)

# def model_train(model, X_train, y_train, X_val, y_val, epochs):
#     model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), batch_size=32, epochs=1000)
#     return model
    


# def evaluate(self, data_dataset):

#     return self.model.evaluate(data_dataset)


# def predict(self, data_dataset):

#     return self.model.predict(data_dataset)

# def model_save(self, model_name):

#     self.model.save(model_name,save_format='tf')

# def model_load(self, model_name):

#     self.model = tf.keras.models.load_model(model_name)    
    
    
    
def load_dataset(base_dir, dataset):
#     dataset = np.array(pd.read_csv(os.path.join(base_dir, dataset+'.csv')))
#     dataset = pd.read_csv(os.path.join(base_dir, dataset+'.csv'))
    dataset = pd.read_csv(f"{base_dir}/{dataset}.csv")
    return dataset
        
def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()
    print (args)
#     train_data, train_labels = _load_training_data(args.train)
#     eval_data, eval_labels = _load_validation_data(args.train)
#     return "123"
    print ("123")
#     train_data = pd.read_csv("./practise/train_data.csv").head(10)
#     train_labels = pd.read_csv("./practise/train_labels.csv").head(10)
#     test_data = pd.read_csv("./practise/test_data.csv").head(10)
#     test_labels = pd.read_csv("./practise/test_labels.csv").head(10)
#     train_data, train_labels = _load_training_data(args.train)
#     eval_data, eval_labels = _load_validation_data(args.train)
    dataset = "sale"
#     print (os.listdir("/opt/ml/processing"))
    sale_train = load_dataset(args.train, "sale_train")
    sale_test = load_dataset(args.test, "sale_test")
    y_train = sale_train[['price_min', 'price_max']].copy()
    X_train = sale_train.drop(['price_min', 'price_max'], axis=1)
    y_val = sale_test[['price_min', 'price_max']].copy()
    X_val = sale_test.drop(['price_min', 'price_max'], axis=1)
    print ("1111")
# #     l = ListingsModel(data)
#     # # l.fill_null_data()
#     train, test = listings_train_test_split(data)
    print ("22222")
#     train_ds = df_to_dataset(train, batch_size=32)
    print ("3333")
#     test_ds = df_to_dataset(test, batch_size=32)
#     y_train = train[['price_min', 'price_max']].copy()
#     X_train = train.drop(['price_min', 'price_max'], axis=1)
#     y_val = test[['price_min', 'price_max']].copy()
#     X_val = test.drop(['price_min', 'price_max'], axis=1)
#     print (data)
    print ("456")
#     feature_layer = feature_columns(data)
#     print (feature_layer)
    print ("789")
    print ("--------------------------")
    print (tf.__version__)
    print ("--------------------------")
#     mdl = model_definition(feature_layer, train_ds, 0.005)
#     mdl = model_definition(feature_layer, dict(X_train), y_train, 0.005)
    mdl = model_definition(X_train, y_train, X_val, y_val, 0.005)
    print ("123123123123123")
#     print (mdl.predict(test_ds))
    print (mdl.predict(X_val))
#     l.model_train(train_ds, 1000)
#     mdl = model_train(mdl, X_train, y_train, X_val, y_val, 1000)
#     mdl = model_definition(np.asarray(X_train).astype('float32'), np.asarray(y_train).astype('float32'), np.asarray(X_val).astype('float32'), np.asarray(y_val).astype('float32'), 1000)
#     print (mdl.predict(np.asarray(X_val).astype('float32')))

#     l.predict(test_ds)
#     l.model_save("listings_"+dataset+"s_model")



#     dataset = "sale"
#     data = load_dataset(args.train, dataset)
#     data = data[['subsector', 'general_size', 'bed', 'bath', 'price_min', 'price_max']]
#     train, test = listings_train_test_split(data)
#     y_train = np.array(train[['price_min', 'price_max']].copy())
#     X_train = np.array(train.drop(['price_min', 'price_max'], axis=1))
#     y_val = np.array(test[['price_min', 'price_max']].copy())
#     X_val = np.array(test.drop(['price_min', 'price_max'], axis=1))
#     print (data)
#     mdl = model_definition(X_train, y_train, X_val, y_val, 0.005)


    print ("456")
#     mdl = model(train_data, train_labels, eval_data, eval_labels)
    print ("789")
    print (args.current_host)
    print (args.hosts[0])
    print (args.hosts)
    mdl.save("s3://sagemaker-us-east-2-476153202769/model/", "mm_model.h5")
    print ("1111")
    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
#         print (args.sm_model_dir)
        mdl.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')