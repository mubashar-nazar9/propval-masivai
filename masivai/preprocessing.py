"""
This file contains the preprocessing steps needed
to be performed, which converts data into useful representation.
For each check or feature engineering, there is a unique
function defined. It takes raw data as an input
and outputs the input features needed to train a model.
"""

# import gin
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
import tensorflow as tf


# @gin.configurable
class Preprocessor:

    def __init__(self, data_file_source):
#         self.dataframe = pd.read_csv(data_file_source)
        pass

    def preprocess_data(self):
        data = self.remove_rows_with_no_labels(self.dataframe, "sale_max_m")

        latitude, longitude = self.get_latitude_and_longitude(data["WKT"])
        data = self.add_latitude_and_longitude_to_dataframe(
            data, latitude, longitude)

        data["area_in"] = self.convert_area_unit_into_marlas(
            data["area_in"])

        data['area_in_marlas'] = self.get_total_marlas_for_each_property(
            data, "area_in", "total_area")

        data = self.fill_missing_values_with_feature_mode(data, column_names=['gc_percentage',
         'height_floor_ft', 'building_line_ft', 'year_of_construction', 'rear_space_ft',
         'side_space_1_ft', 'side_space_2_ft', 'block_subsector_name', 'area_in_marlas'])

        features_to_be_used = ['block_subsector_name', 'house_vacant', 'gc_percentage', 'height_floor_ft',
         'building_line_ft', 'year_of_construction', 'rear_space_ft', 'side_space_2_ft', 'side_space_1_ft',
         'parking_provision', 'far', 'street_width_ft', 'number_of_stories', 'grading_a_b_c', 'area_in_marlas',
         'latitude', 'longitude']

        data[features_to_be_used[13]] = self.assign_labels_to_categorical_data(
            data[features_to_be_used[13]])

        data[features_to_be_used[1]] = self.fill_white_spaces_with_zero(
            data[features_to_be_used[1]])
        data[features_to_be_used[11]] = self.fill_white_spaces_with_zero(
            data[features_to_be_used[11]])
        data[features_to_be_used[12]] = self.fill_white_spaces_with_zero(
            data[features_to_be_used[12]])

        data[features_to_be_used[9]] = self.replace_text_values_with_numerical_values(
            data[features_to_be_used[9]], find=["Yes"], replace=[1])

        block_subsector_name = self.one_hot_encode_categorical_features(
            data[features_to_be_used[0]])
        is_house_vacant = self.one_hot_encode_categorical_features(
            data[features_to_be_used[1]])

        processed_dataset = self.combine_sub_dataframes(
            [block_subsector_name, is_house_vacant, data[features_to_be_used[2:]]])

        processed_dataset = self.convert_dataframe_to_float_values(processed_dataset)

        return processed_dataset

    def remove_rows_with_no_labels(self, dataframe, column_name):
        """
        Only rows with filled label values are kept
        """
        return dataframe[dataframe[column_name].notnull()]

    def get_latitude_and_longitude(self, feature_WKT):
        """
        Longitude and latitude values are extracted
        """
        latitude, longitude = [], []
        for i in feature_WKT:
            latitude_i = i.split("(")[3].split(" ")[1]
            longitude_i = i.split("(")[3].split(" ")[0]
            latitude.append(latitude_i)
            longitude.append(longitude_i)

        return latitude, longitude

    def add_latitude_and_longitude_to_dataframe(self, dataframe, latitude, longitude):
        dataframe["latitude"] = latitude
        dataframe["longitude"] = longitude

        return dataframe

    def convert_area_unit_into_marlas(self, area_unit):
        """
        Area (Marla, Kanal, Acre) is transformed into marlas
        """
        area_unit.loc[area_unit == "Marla"] = 1
        area_unit.loc[area_unit == "Kanal"] = 20
        area_unit.loc[area_unit == "Acre"] = 160

        return area_unit

    def get_total_marlas_for_each_property(self, dataframe, area_unit, total_area):
        """
        This will take two columns (area_in, total_area) as input and create
        new column (area (marla)), which will have the marla count
        """
        return dataframe[area_unit]*dataframe[total_area]

    def fill_missing_values_with_feature_mode(self, dataframe, column_names):
        """
        All missing values are filled with mode of their columns
        """
        for column_name in column_names:
            mode = dataframe[column_name].mode()
            dataframe[column_name].fillna(mode[0], inplace=True)

        return dataframe

    def assign_labels_to_categorical_data(self, feature_grading_a_b_c):
        """
        Assigns labels to categories
        """
        le = preprocessing.LabelEncoder()
        le.fit(feature_grading_a_b_c)
        return le.transform(feature_grading_a_b_c)

    def fill_white_spaces_with_zero(self, feature):
        """
        Replace empty feature values with numbers
        """
        feature.loc[feature == ' '] = 0
        return feature

    def replace_text_values_with_numerical_values(self, parking_provision, find, replace):
        """
        Text values are replaced with appropriate numerical values
        input (feature, find (list of words to be replaced), replace (list of numbers to be placed wrt find))
        Yes would be replace with 1, No with 0
        """
        for i in range(len(find)):
            parking_provision.loc[parking_provision == find[i]] = replace[i]
        return parking_provision
    
    def one_hot_encode_categorical_features(self, feature_name):
        """
        One hot encoding is done for categorical features
        """
        return pd.get_dummies(feature_name)

    def combine_sub_dataframes(self, dataframes_list):
        """
        Combines all sub dataframes
        """
        dataframe_0 = dataframes_list[0]
        for i in range(1, len(dataframes_list)):
            dataframe_1 = dataframes_list[i]
            dataframe_0 = pd.concat([dataframe_0, dataframe_1], axis=1)
        return dataframe_0

    def convert_dataframe_to_float_values(self, dataframe):
        """
        This will convert all dataframe individual indices to float
        """
        return dataframe.astype(float)

    def convert_currency_values_in_lac_to_multiples_of_thousand(self, currency_column):
        
        """
        Convert currency values to their respective numerical format
        """
        for i in currency_column.index:
            try:
                int(currency_column[i])
            except:
                currency_value_split = currency_column[i].split()
                if currency_value_split[1] == 'Lac':
                    currency_column[i] = int(currency_value_split[0]) * 100
        
        return currency_column

    #listings_preprocessing_functions
    def preprocess_listings_data(self, properties, areas, cities):
        data = self.dataframes_merge(properties, areas, cities)
        data = self.columns_selection(data)
        data = self.rename_columns(data)
        data = self.filter_relevant_columns(data)
        data = self.filtered_sales_data(data)
        sectors = self.list_possible_sectors()
        sectors_index, index = self.fetch_each_sector_listings(data, sectors)
        data = self.valid_data_filtered(data, index)
        data = self.adding_subsector_level_information_to_valid_data(data, sectors_index, sectors)
        data = self.remove_duplicates(data)
        data = self.filtered_columns(data, ["purpose", "type", "subtype", "subsector", "general_size", "bed", "bath", "price"])
        data = self.split_columns(data, "subsector", ["sector", "subsector"])
        data = self.fill_null_data(data)
        data = self.remove_invalid_prices(data)

        #unique rows still prending, price min max range (original)
        data = self.min_max_range(data)
        data.fillna(0, inplace=True)
        sale = data[data.purpose=="sale"][['type', 'subtype', 'subsector', 'general_size', 'bed', 'bath', 'price_min', 'price_max', 'sector']]
        rent = data[data.purpose=="rent"][['type', 'subtype', 'subsector', 'general_size', 'bed', 'bath', 'price_min', 'price_max', 'sector']]
        
        return sale, rent
    
    #merging data with area and city
    def dataframes_merge(self, properties, areas, cities):
        temp_1 = pd.merge(properties, areas, left_on="area_id", right_on="id", how="left")
        data = pd.merge(temp_1, cities, left_on="city_id_x", right_on="id", how="left")
        return data

    #selecting only important columns
    def columns_selection(self, data):
        data = data[["id_x", "purpose", "type", "subtype", "address", "lat_x", "lon_x", "description_x", "price", "size_x", "size_unit", "general_size", "bed", "bath", "features", "created_at_x", "updated_at_x", "custom_title_x", "name_x", "lat_y", "lon_y", "name_y"]]
        return data

    #renaming columns
    def rename_columns(self, data):
        data.rename(columns={"id_x":"id", "lat_x":"property_lat", "lon_x":"property_lon", "description_x": "description", "size_x":"size",  "created_at_x":"created_at", "updated_at_x":"updated_at", "custom_title_x":"custom_title", "name_x":"area_name", "lat_y":"area_lat", "lon_y":"area_lon", "name_y":"city_name"}, inplace=True)
        return data

    def filter_relevant_columns(self, data):
        data = data[['id', 'purpose', 'type', 'subtype', 'address', 'description', 'price', 'size', 'size_unit',
               'general_size', 'bed', 'bath', 'features', 'created_at', 'updated_at',
               'custom_title', 'area_name', 'area_lat', 'area_lon', 'city_name']]#.to_csv("Filtered Data (EDA).csv", index=False)
        return data
    
    """filtering only islambad data"""
    def filtered_sales_data(self, data):
        islamabad_sales = data[(data["city_name"] == "Islamabad")]
        return islamabad_sales
    
    """possible sectors combinations in islamabad"""
    def list_possible_sectors(self):
        sector = ['e', 'f', 'g', 'h', 'i']
        number = np.arange(20)
        sub_sector = [1,2,3,4]

        sectors = []
        for i in sector:
            for j in number:
                for k in sub_sector:
                    sectors.append(i + "-" + str(j) + "/" + str(k))

        return sectors
    
    """getting indexes of islamabad sectors found in listings"""
    def fetch_each_sector_listings(self, islamabad_sales, sectors):
        index = []
        arr_list = []
        sec_index = []
        for i in sectors:
            print (str(i) + ": " + str(islamabad_sales[(islamabad_sales["address"].str.lower().str.contains(i)) |  (islamabad_sales["description"].str.lower().str.contains(i))  | (islamabad_sales["area_name"].str.lower().str.contains(i)) | (islamabad_sales["custom_title"].str.lower().str.contains(i))].shape[0]))
            arr_list.append(islamabad_sales[(islamabad_sales["address"].str.lower().str.contains(i)) |  (islamabad_sales["description"].str.lower().str.contains(i))  | (islamabad_sales["area_name"].str.lower().str.contains(i)) | (islamabad_sales["custom_title"].str.lower().str.contains(i))].shape[0])
            sec_index.append(islamabad_sales[(islamabad_sales["address"].str.lower().str.contains(i)) |  (islamabad_sales["description"].str.lower().str.contains(i))  | (islamabad_sales["area_name"].str.lower().str.contains(i)) | (islamabad_sales["custom_title"].str.lower().str.contains(i))])
            index.extend(islamabad_sales[(islamabad_sales["address"].str.lower().str.contains(i)) |  (islamabad_sales["description"].str.lower().str.contains(i))  | (islamabad_sales["area_name"].str.lower().str.contains(i)) | (islamabad_sales["custom_title"].str.lower().str.contains(i))].index)

        sectors_index = []
        for i in sec_index:
            sectors_index.append(i.index)

        return sectors_index, index
    
    """islamabad accurate sectors data filtered"""
    def valid_data_filtered(self, islamabad_sales, index):
        filtered_islamabad_sales = islamabad_sales.loc[index]
        return filtered_islamabad_sales
    
    """separate column subsector level info e.g. e-7/2"""
    def adding_subsector_level_information_to_valid_data(self, filtered_islamabad_sales, sectors_index, sectors):
        filtered_islamabad_sales["subsector"] = 0
        for i in range(len(sectors_index)):
            if len(sectors_index[i])>0:
                for j in sectors_index[i]:
                    filtered_islamabad_sales.loc[j, "subsector"]=sectors[i]
        return filtered_islamabad_sales
        
    """duplicate rows removed"""
    def remove_duplicates(self, filtered_islamabad_sales):
        filtered_islamabad_sales.drop_duplicates(inplace=True)
        return filtered_islamabad_sales
    
    """specfic columns filter out for model building"""
    def filtered_columns(self, filtered_islamabad_sales, columns):
        filtered_islamabad_sales = filtered_islamabad_sales[columns]
        return filtered_islamabad_sales
    
    def split_columns(self, filtered_islamabad_sales, column, converted_columns):
        filtered_islamabad_sales[converted_columns]=filtered_islamabad_sales[column].str.split("/", expand=True)
        return filtered_islamabad_sales
    
    def fill_null_data(self, data):
        data = data.fillna(0)
        return data
    
    def remove_invalid_prices(self, data):
        data = data[data.price!=0]
        return data
    
    """exclude low/inappropriate prices"""
    def filtered_prices(self, data, filter_check):
        data = data[data.price>filter_check]
        return data
    
    def min_max_range(self, data):
        minimum = data.price*0.9
        maximum = data.price*1.1
        data.pop("price")
        data["price_min"] = minimum
        data["price_max"] = maximum
        return data
    
    def tf_feature_layer(self, data):
        numerical_column = ["general_size", "bed", "bath"]
        #         categorical_indicator_column = ["subsector", "purpose"]
        categorical_indicator_column = ["subsector"]
        categorical_embedding_column = ["sector", "type", "subtype"]

        feature_columns = []

        for header in numerical_column:
            feature_columns.append(feature_column.numeric_column(header))

        for feature_name in categorical_indicator_column:
            vocabulary = data[feature_name].unique()
            cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
            one_hot = feature_column.indicator_column(cat_c)
            feature_columns.append(one_hot)

        for feature_name in categorical_embedding_column:
            vocabulary = data[feature_name].unique()
            cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
            embeding = feature_column.embedding_column(cat_c, dimension=4)
            feature_columns.append(embeding)

        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        return feature_layer
#         feature_layer(dict(sale)).numpy().shape

    def listings_train_test_split(self, input_data):

        train, test = train_test_split(input_data, test_size=0.2)
        return train, test
    
# gin.parse_config_file('../config.gin')

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    
    
#     df = pd.read_csv(
#         f"{base_dir}/input/abalone-dataset.csv",
#         header=None 
# #         names=feature_columns_names + [label_column],
# #         dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
#     )
####################################################
###without glue job####
    properties = pd.read_csv(
        f"{base_dir}/input/properties.csv"
    )
    

    cities = pd.read_csv(
        f"{base_dir}/input/cities.csv"
    )
    
    
    areas = pd.read_csv(
        f"{base_dir}/input/areas.csv"
    )
    
    p = Preprocessor(None)
    sale, rent = p.preprocess_listings_data(properties, areas, cities)
############################################################

############################################
#glue job#####
    sale = pd.read_csv(
        f"{base_dir}/input/sale.csv"
    )
    
    rent = pd.read_csv(
        f"{base_dir}/input/rent.csv"
    )
###########################################
    
    sale.fillna(0, inplace=True)
    rent.fillna(0, inplace=True)
    
    sale = p.min_max_range(sale)
    rent = p.min_max_range(rent)
    
    
    sale_train, sale_test = p.listings_train_test_split(sale)
    rent_train, rent_test = p.listings_train_test_split(rent)
    
#     numerical_column = ["general_size", "bed", "bath", "price_min", "price_max"]
#     sale_train = sale_train[numerical_column]
#     sale_test = sale_test[numerical_column]
#     rent_train = rent_train[numerical_column]
#     rent_test = rent_test[numerical_column]
    
    
    sale_feature_layer = p.tf_feature_layer(sale_train)
    rent_feature_layer = p.tf_feature_layer(rent_train)

    sale_train_ft = pd.DataFrame(sale_feature_layer(dict(sale_train)).numpy())
    sale_train.reset_index(drop=True, inplace=True)
    sale_train_ft["price_min"] = sale_train["price_min"]
    sale_train_ft["price_max"] = sale_train["price_max"]
    sale_test_ft = pd.DataFrame(sale_feature_layer(dict(sale_test)).numpy())
    sale_test.reset_index(drop=True, inplace=True)
    sale_test_ft["price_min"] = sale_test["price_min"]
    sale_test_ft["price_max"] = sale_test["price_max"]

    rent_train_ft = pd.DataFrame(rent_feature_layer(dict(rent_train)).numpy())
    rent_train.reset_index(drop=True, inplace=True)
    rent_train_ft["price_min"] = rent_train["price_min"]
    rent_train_ft["price_max"] = rent_train["price_max"]
    rent_test_ft = pd.DataFrame(rent_feature_layer(dict(rent_test)).numpy())
    rent_test.reset_index(drop=True, inplace=True)
    rent_test_ft["price_min"] = rent_test["price_min"]
    rent_test_ft["price_max"] = rent_test["price_max"]

    print (sale_train.head())
#     sale_train.to_csv(f"{base_dir}/train/sale_train.csv", index=False)
#     sale_test.to_csv(f"{base_dir}/test/sale_test.csv", index=False)
    sale_train_ft.to_csv(f"{base_dir}/train/sale_train.csv", index=False)
    sale_test_ft.to_csv(f"{base_dir}/test/sale_test.csv", index=False)
#     rent_train_ft.to_csv(f"{base_dir}/rent/rent_train.csv", index=False)
#     rent_test_ft.to_csv(f"{base_dir}/rent/rent_test.csv", index=False)

    print (sale_feature_layer)
#     tf.keras.models.save_model(sale_feature_layer, filepath="./dataset/sale__layer")
#     tf.keras.models.save_model(sale_feature_layer, filepath=f"{base_dir}/sale/sale_ft_layer")
#     tf.keras.models.save_model(rent_feature_layer, filepath=f"{base_dir}/rent/rent_ft_layer")
    
#     tf.keras.models.save_model(sale_feature_layer, filepath="./dataset/sale_ft_layer")
#     tf.keras.models.save_model(rent_feature_layer, filepath="./dataset/rent_ft_layer")