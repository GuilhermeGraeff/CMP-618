import tensorflow as tf

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

    dataset_file_path = tf.keras.utils.get_file('../dados/csv/'+type+'.csv', DATASET_URL)

    dataset = tf.data.experimental.make_csv_dataset(
      dataset_file_path,
      batch_size=32,
      label_name='samples',
      na_value="?",
      num_epochs=100,
      ignore_errors=True)
    
    model_size = dataset.cardinality().numpy()
    
    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset, left_size=0.7, shuffle=True)

    network = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(model_size,)), 
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(n_classes),
        tf.keras.layers.Softmax()
    ])

    network.compile(optimizer='adamw',                        
                loss='sparse_categorical_crossentropy',
                metrics=['f1score'])
    
    network.fit(train_dataset, epochs=100)
    loss, f1score = network.evaluate(test_dataset)

    with open('../models/results.txt', 'a') as file:
        file.write(f'{type}: Loss = {loss * 100}; F1 Score = {f1score * 100}\n')
    
    network.save('../models/'+type+'.keras')