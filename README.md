## GitHub README: Building a Comment Toxicity Model with Deep Learning in Python

---

### **Table of Contents**
1. [Initial Setup and Data Preparation](#part-1-initial-setup-and-data-preparation)
2. [Prepare Comments](#part-2-prepare-comments)
3. [Build a Deep Learning Model](#part-3-build-a-deep-learning-model)
4. [Making Predictions](#part-4-making-predictions)
5. [Evaluate the Model](#part-5-evaluate-the-model)
6. [Build a Deep Learning Gradio App](#part-6-build-a-deep-learning-gradio-app)

---

### Part 1: Initial Setup and Data Preparation
We'll start by preparing our environment and data for building a machine-learning model that identifies toxicity in comments. The dataset is in a CSV file, containing comments tagged with labels like "toxic", "threatening", and "identity hate". Our process involves tokenizing these comments and creating embeddings for LSTM (Long Short-Term Memory) networks and additional layers.

After building and training, we serialize the model to an H5 file, preserving its architecture and learned weights for future use.

#### Coding Instructions:
- Install libraries (TensorFlow, NumPy, Pandas, Matplotlib, Scikit-learn):
  ```bash
  !pip install tensorflow tensorflow-gpu pandas matplotlib sklearn
  ```
- Import libraries:
  ```python
  import pandas as pd
  import tensorflow as tf
  import numpy as np
  ```
- Load data into a DataFrame:
  ```python
  df = pd.read_csv("train.csv")
  ```

---

### Part 2: Prepare Comments
We tokenize and vectorize comments using Keras' `TextVectorization`. After splitting data into features (X) and targets (y), we adapt the vectorizer to our data and tokenize all comments.

We create an efficient data pipeline using TensorFlow's functions like `cache`, `shuffle`, `batch`, and `prefetch`. The dataset is then split into training, validation, and test sets.

#### Key Steps:
- Initialize the vectorizer:
  ```python
  from tensorflow.keras.layers import TextVectorization
  MAX_FEATURES = 200000
  vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
  ```
- Tokenize the comments and create the data pipeline:
  ```python
  X = df["comments_text"]
  y = df[df.columns[2:]].values
  vectorizer.adapt(X.values)
  vectorized_text = vectorizer(X.values)
  # Data pipeline steps
  ```

---

### Part 3: Build a Deep Learning Model
This part involves building a model using Keras. We start with an embedding layer, followed by a bidirectional LSTM layer, and several dense layers. The output layer has 6 units with sigmoid activation for binary classification. The model is compiled with `BinaryCrossentropy` and `Adam` optimizer.

#### Model Construction:
- Import necessary layers and initialize the model:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
  model = Sequential()
  ```
- Add layers and compile the model:
  ```python
  model.add(Embedding(MAX_FEATURES+1, 32))
  model.add(Bidirectional(LSTM(32, activation='tanh')))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(6, activation='sigmoid'))
  model.compile(loss='BinaryCrossentropy', optimizer='Adam')
  ```
- Train and visualize training history:
  ```python
  history = model.fit(train, epochs=1, validation_data=val)
  ```

---

### Part 4: Making Predictions
This section explains how to use the trained model to predict the toxicity of new comments. The process involves vectorizing the input text, reshaping it for the model, and interpreting the model's predictions.

#### Prediction Process:
- Prepare and reshape input text for prediction.
- Use `model.predict` to get prediction probabilities.
- Apply a threshold to interpret predictions.
- Batch prediction steps are also detailed.

---

### Part 5: Evaluate the Model
We assess model performance using precision, recall, and categorical accuracy. The evaluation involves iterating over test batches, updating metrics, and calculating final values.

#### Evaluation Steps:
- Import and instantiate metrics:
  ```python
  from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
  ```
- Iterate over test batches and update metrics.
- Display final metric values.

---

### Part 6: Build a Deep Learning Gradio App
This final part guides on integrating the model into a Gradio app for interactive predictions. Steps include installation, model saving and loading, and building the Gradio interface.

#### Gradio App Development:
- Install Gradio and dependencies:
  ```bash
  !pip install gr

adio jinja2
  ```
- Build and launch the Gradio interface:
  ```python
  import gradio as gr
  # Model saving, loading, and interface building steps
  ```

---

### **Conclusion**
This tutorial provides a comprehensive guide to building a deep learning model for comment toxicity detection using Python and TensorFlow. It covers everything from data preparation to deploying a Gradio web app for user interaction. Experimentation and feedback are encouraged to enhance model performance.

[GitHub Repository](https://github.com/KhalidOvrFitter/CommentToxicityModel)
