# Comment Toxicity Detection with Deep Learning

This project aims to build a deep learning model to identify toxic comments using Python. We'll work with a labeled dataset, tokenize the comments, and feed them into an LSTM network.

## Setup & Data Prep

We start by installing required libraries:

```bash
!pip install tensorflow tensorflow-gpu pandas matplotlib sklearn
```

Load libraries and the dataset:

```python
import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv("train.csv")
```

## Tokenization & Vectorization

Using Keras' `TextVectorization`:

```python
from tensorflow.keras.layers import TextVectorization

MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800)
X = df["comments_text"]
y = df.iloc[:, 2:].values

vectorizer.adapt(X.values)
```

Prepare data pipeline:

```python
dataset = tf.data.Dataset.from_tensor_slices((vectorizer(X.values), y))
dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)

# Split the data
train_dataset = dataset.take(int(len(dataset)*.7))
val_dataset = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test_dataset = dataset.skip(int(len(dataset)*.9))
```

## Model Building

Define and compile the model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional

model = Sequential([
    Embedding(MAX_FEATURES+1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])

model.compile(loss='BinaryCrossentropy', optimizer='Adam')
```

Train and evaluate:

```python
model.summary()
history = model.fit(train_dataset, epochs=1, validation_data=val_dataset)
```

Visualize training:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()
```

## Predictions

To make predictions:

```python
input_text = vectorizer("Example text here")
model_input = np.expand_dims(input_text, axis=0)
prediction = model.predict(model_input)

# Batch predictions
batch_x, batch_y = test_dataset.as_numpy_iterator().next()
batch_predictions = model.predict(batch_x)
```

Interpret and analyze model output for informed adjustments and improvements.

**Important**: For best results, experiment with more epochs, data, and tuning based on model performance.
