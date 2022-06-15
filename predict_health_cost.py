import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


dataset = pd.read_csv('insurance.csv')
dataset.tail()

dataset["sex"] = dataset["sex"].map({"male": 0, "female": 1})
dataset["smoker"] = dataset["smoker"].map({"no": 0, "yes": 1})
dataset["region"] = dataset["region"].map({"northwest": 0, "northeast": 1, 
                                           "southwest": 2, "southeast": 3,
                                           "midwest": 4, "mideast": 5})

print(dataset)


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset["expenses"]
print(train_labels)
test_labels = test_dataset["expenses"]

train_dataset.drop('expenses', 1, inplace=True)
test_dataset.drop('expenses', 1, inplace=True)


model = keras.Sequential()
model.add(layers.Dense(75, activation='relu'))
model.add(layers.Dense(25, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1))

model.compile(tf.optimizers.Adam(learning_rate=0.01), 
              loss=tf.losses.mean_absolute_error,
              metrics=[tf.keras.metrics.mae, tf.keras.metrics.mse])

model.fit(train_dataset, train_labels, epochs=100, verbose=1)


# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
