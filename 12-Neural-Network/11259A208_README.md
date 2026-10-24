# 12. Neural Network

[Back to experiments](../README.md)

A supervised binary classifier predicts Pass or Fail from hours studied, attendance, and assignments. The network has hidden layers of 8 and 4 ReLU units and one sigmoid output. It trains for 50 epochs with Adam and binary cross-entropy.

[Python source](neural_network.py) | [Dataset](data.csv) | [Full training output](output.txt)

## Code

```python
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

tf.keras.utils.set_random_seed(42)

data = pd.read_csv("data.csv")

X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

y = y.map({'Fail': 0, 'Pass': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=2
)

y_pred = model.predict(X_test, verbose=0)
y_pred = (y_pred > 0.5).astype(int).ravel()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Actual training output

```text
Epoch 1/50
1/1 - 1s - 1s/step - accuracy: 0.8333 - loss: 0.5944 - val_accuracy: 1.0000 - val_loss: 0.5983
Epoch 2/50
1/1 - 0s - 89ms/step - accuracy: 0.8333 - loss: 0.5917 - val_accuracy: 1.0000 - val_loss: 0.5967
Epoch 3/50
1/1 - 0s - 89ms/step - accuracy: 0.8333 - loss: 0.5891 - val_accuracy: 1.0000 - val_loss: 0.5951
Epoch 4/50
1/1 - 0s - 87ms/step - accuracy: 0.8333 - loss: 0.5865 - val_accuracy: 1.0000 - val_loss: 0.5936
Epoch 5/50
1/1 - 0s - 92ms/step - accuracy: 0.8333 - loss: 0.5840 - val_accuracy: 1.0000 - val_loss: 0.5920
Epoch 6/50
1/1 - 0s - 90ms/step - accuracy: 0.9167 - loss: 0.5814 - val_accuracy: 1.0000 - val_loss: 0.5905
Epoch 7/50
1/1 - 0s - 88ms/step - accuracy: 0.9167 - loss: 0.5789 - val_accuracy: 1.0000 - val_loss: 0.5891
Epoch 8/50
1/1 - 0s - 90ms/step - accuracy: 0.9167 - loss: 0.5763 - val_accuracy: 1.0000 - val_loss: 0.5877
Epoch 9/50
1/1 - 0s - 90ms/step - accuracy: 0.9167 - loss: 0.5737 - val_accuracy: 1.0000 - val_loss: 0.5864
Epoch 10/50
1/1 - 0s - 90ms/step - accuracy: 0.9167 - loss: 0.5710 - val_accuracy: 1.0000 - val_loss: 0.5851
Epoch 11/50
1/1 - 0s - 85ms/step - accuracy: 0.9167 - loss: 0.5683 - val_accuracy: 1.0000 - val_loss: 0.5837
Epoch 12/50
1/1 - 0s - 93ms/step - accuracy: 0.9167 - loss: 0.5656 - val_accuracy: 1.0000 - val_loss: 0.5824
Epoch 13/50
1/1 - 0s - 100ms/step - accuracy: 0.9167 - loss: 0.5629 - val_accuracy: 1.0000 - val_loss: 0.5811
Epoch 14/50
1/1 - 0s - 97ms/step - accuracy: 0.9167 - loss: 0.5603 - val_accuracy: 1.0000 - val_loss: 0.5798
Epoch 15/50
1/1 - 0s - 93ms/step - accuracy: 0.9167 - loss: 0.5576 - val_accuracy: 1.0000 - val_loss: 0.5785
Epoch 16/50
1/1 - 0s - 85ms/step - accuracy: 0.9167 - loss: 0.5550 - val_accuracy: 1.0000 - val_loss: 0.5772
Epoch 17/50
1/1 - 0s - 86ms/step - accuracy: 0.9167 - loss: 0.5523 - val_accuracy: 1.0000 - val_loss: 0.5759
Epoch 18/50
1/1 - 0s - 86ms/step - accuracy: 0.9167 - loss: 0.5497 - val_accuracy: 1.0000 - val_loss: 0.5746
Epoch 19/50
1/1 - 0s - 88ms/step - accuracy: 0.9167 - loss: 0.5470 - val_accuracy: 1.0000 - val_loss: 0.5733
Epoch 20/50
1/1 - 0s - 86ms/step - accuracy: 1.0000 - loss: 0.5447 - val_accuracy: 1.0000 - val_loss: 0.5719
Epoch 21/50
1/1 - 0s - 90ms/step - accuracy: 1.0000 - loss: 0.5424 - val_accuracy: 1.0000 - val_loss: 0.5704
Epoch 22/50
1/1 - 0s - 87ms/step - accuracy: 1.0000 - loss: 0.5402 - val_accuracy: 1.0000 - val_loss: 0.5689
Epoch 23/50
1/1 - 0s - 87ms/step - accuracy: 1.0000 - loss: 0.5379 - val_accuracy: 1.0000 - val_loss: 0.5673
Epoch 24/50
1/1 - 0s - 92ms/step - accuracy: 1.0000 - loss: 0.5356 - val_accuracy: 1.0000 - val_loss: 0.5657
Epoch 25/50
1/1 - 0s - 91ms/step - accuracy: 1.0000 - loss: 0.5333 - val_accuracy: 1.0000 - val_loss: 0.5640
Epoch 26/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.5310 - val_accuracy: 1.0000 - val_loss: 0.5623
Epoch 27/50
1/1 - 0s - 90ms/step - accuracy: 1.0000 - loss: 0.5287 - val_accuracy: 1.0000 - val_loss: 0.5606
Epoch 28/50
1/1 - 0s - 89ms/step - accuracy: 1.0000 - loss: 0.5264 - val_accuracy: 1.0000 - val_loss: 0.5588
Epoch 29/50
1/1 - 0s - 84ms/step - accuracy: 1.0000 - loss: 0.5241 - val_accuracy: 1.0000 - val_loss: 0.5571
Epoch 30/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.5218 - val_accuracy: 1.0000 - val_loss: 0.5553
Epoch 31/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.5196 - val_accuracy: 1.0000 - val_loss: 0.5535
Epoch 32/50
1/1 - 0s - 88ms/step - accuracy: 1.0000 - loss: 0.5173 - val_accuracy: 1.0000 - val_loss: 0.5517
Epoch 33/50
1/1 - 0s - 88ms/step - accuracy: 1.0000 - loss: 0.5150 - val_accuracy: 1.0000 - val_loss: 0.5498
Epoch 34/50
1/1 - 0s - 87ms/step - accuracy: 1.0000 - loss: 0.5127 - val_accuracy: 1.0000 - val_loss: 0.5480
Epoch 35/50
1/1 - 0s - 84ms/step - accuracy: 1.0000 - loss: 0.5104 - val_accuracy: 1.0000 - val_loss: 0.5462
Epoch 36/50
1/1 - 0s - 87ms/step - accuracy: 1.0000 - loss: 0.5081 - val_accuracy: 1.0000 - val_loss: 0.5443
Epoch 37/50
1/1 - 0s - 93ms/step - accuracy: 1.0000 - loss: 0.5058 - val_accuracy: 1.0000 - val_loss: 0.5424
Epoch 38/50
1/1 - 0s - 97ms/step - accuracy: 1.0000 - loss: 0.5035 - val_accuracy: 1.0000 - val_loss: 0.5406
Epoch 39/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.5013 - val_accuracy: 1.0000 - val_loss: 0.5387
Epoch 40/50
1/1 - 0s - 90ms/step - accuracy: 1.0000 - loss: 0.4990 - val_accuracy: 1.0000 - val_loss: 0.5368
Epoch 41/50
1/1 - 0s - 88ms/step - accuracy: 1.0000 - loss: 0.4967 - val_accuracy: 1.0000 - val_loss: 0.5349
Epoch 42/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.4945 - val_accuracy: 1.0000 - val_loss: 0.5330
Epoch 43/50
1/1 - 0s - 88ms/step - accuracy: 1.0000 - loss: 0.4922 - val_accuracy: 1.0000 - val_loss: 0.5311
Epoch 44/50
1/1 - 0s - 86ms/step - accuracy: 1.0000 - loss: 0.4899 - val_accuracy: 1.0000 - val_loss: 0.5292
Epoch 45/50
1/1 - 0s - 89ms/step - accuracy: 1.0000 - loss: 0.4877 - val_accuracy: 1.0000 - val_loss: 0.5273
Epoch 46/50
1/1 - 0s - 93ms/step - accuracy: 1.0000 - loss: 0.4854 - val_accuracy: 1.0000 - val_loss: 0.5254
Epoch 47/50
1/1 - 0s - 90ms/step - accuracy: 1.0000 - loss: 0.4832 - val_accuracy: 1.0000 - val_loss: 0.5235
Epoch 48/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.4810 - val_accuracy: 1.0000 - val_loss: 0.5216
Epoch 49/50
1/1 - 0s - 85ms/step - accuracy: 1.0000 - loss: 0.4787 - val_accuracy: 1.0000 - val_loss: 0.5198
Epoch 50/50
1/1 - 0s - 86ms/step - accuracy: 1.0000 - loss: 0.4766 - val_accuracy: 1.0000 - val_loss: 0.5179
Accuracy: 1.0
```

## Result

Test accuracy: **100% (4 of 4 held-out records)**. This is a very small dataset; the result describes this split. Random seed: 42.

## Run

Install the root requirements, then run `python neural_network.py` from this experiment folder. To save the epoch log, run `python neural_network.py > output.txt`.
