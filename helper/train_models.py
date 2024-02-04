import tensorflow as tf
import pandas as pd
import math
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def df_to_dataset(dataframe, shuffle=True):
    dataframe = dataframe.copy()

    batch_size = math.floor(dataframe.shape[0] / 10)

    dataframe.pop('samples')

    labels = dataframe.pop('type')

    label_encoder = LabelEncoder()
    vec = label_encoder.fit_transform(labels)

    labels = tf.keras.utils.to_categorical(y=vec)

    dataset = np.array(dataframe)

    #dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

    #if shuffle:
    #    dataset = dataset.shuffle(buffer_size=len(dataframe))

     #   dataset = dataset.batch(batch_size)

    return dataset, labels

def f(X):
    return regression.predict([X[:, i] for i in range(X.shape[1])]).flatten()

with open('../dados/data_links.txt') as file:
    next(file)
    data_links = file.readlines()

datasets = {}
for link in data_links:
    cols = link.split()
    datasets[cols[0]] = [cols[1], cols[2]]

for type, data in datasets.items():
    DATASET_URL = data[0]
    n_classes = int(data[1])

    dataframe_file_path = tf.keras.utils.get_file(fname='/home/aschoier/CMP-618/dados/csv/'+type+'.csv', origin=DATASET_URL)

    dataframe = pd.read_csv(dataframe_file_path)

    train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.2)
    train_dataframe, val_dataframe = train_test_split(train_dataframe, test_size=0.2)

    dataset = df_to_dataset(dataframe)

    train_dataset, train_labels = df_to_dataset(train_dataframe)
    test_dataset, test_labels = df_to_dataset(test_dataframe, False)
    val_dataset, val_labels = df_to_dataset(val_dataframe, False)
 
    cols = list(dataframe)

    cols.pop(0)
    cols.pop(1)

    feature_columns = []

    for col_name in cols:
        feature_columns.append(tf.feature_column.numeric_column(col_name))

    network = tf.keras.Sequential([
#        tf.keras.layers.DenseFeatures(feature_columns),
        tf.keras.layers.Input((len(dataframe.columns)-2,)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(n_classes),
        tf.keras.layers.Softmax()
    ])

    network.compile(optimizer='adam',                        
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    network.fit(x=train_dataset, y=train_labels, epochs=100, validation_data=(val_dataset, val_labels), verbose=2)
    loss, accuracy = network.evaluate(x=test_dataset, y=test_labels)

    with open('/home/aschoier/CMP-618/models/results.txt', 'a') as file:
        file.write(f'{type}: loss = {loss}; accuracy = {accuracy}\n')
    
    network.save('../models/'+type+'.keras')

    explainer = shap.KernelExplainer(f, val_dataset)

    explainer = shap.DeepExplainer(model, background)

    shap_values = e.shap_values(test_dataset)

    shap_values = explainer.shap_values(train_dataset, nsamples=500)

    plot = shap.force_plot(explainer.expected_value, shap_values, train_dataset, show=False) 

    print(plot)
