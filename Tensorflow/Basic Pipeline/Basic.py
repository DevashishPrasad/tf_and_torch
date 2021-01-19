### Split the dataset

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=45)

X_train=None; X_test=None; Y_train=None; Y_test=None;

for train_index, test_index in skf.split(x, y):
    X_train = x[train_index]
    X_test = x[test_index]
    Y_train = y[train_index]
    Y_test = y[test_index]
    break

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

## Build and train Model

import tensorflow as tf
from tensorflow import keras

### Create TF datasets

def create_dataset(xs, ys, n_classes=10):
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(ys)).batch(32)

train_dataset = create_dataset(X_train, Y_train, n_classes=5)
test_dataset = create_dataset(X_test, Y_test, n_classes=5)

### Define Model

def create_model():
  model = keras.Sequential([
      keras.layers.Reshape(target_shape=(20 * 39,), input_shape=(20, 39)),
      keras.layers.Dense(units=128, activation='relu'),
      keras.layers.Dense(units=5, activation='softmax')
  ])

  model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  
  return model

model = create_model()
model.summary()

### Train

history = model.fit(
    train_dataset, 
    epochs=10, 
    validation_data=test_dataset
)

### Save and load model

# Save the weights
model.save_weights('/content/drive/MyDrive/Freelance/Models/5_ex')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('/content/drive/MyDrive/Freelance/Models/5_ex')

## Visualize Training

from matplotlib import pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
