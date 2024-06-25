# Toxic Comment Classification on Social Media

## Project Overview

This project aims to build a model to classify toxic comments on social media using a dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). The dataset includes various types of toxicity such as:
- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

The objective is to predict the probability of each type of toxicity for a given comment. This notebook details the process from data preprocessing to model training, evaluation, and deployment.

## Dataset Description

The dataset contains Wikipedia comments that have been labeled by human raters for toxic behavior. Each comment can have multiple labels indicating the type of toxicity. The dataset is divided into the following columns:
- **id**: Unique identifier for each comment.
- **comment_text**: The actual comment text.
- **toxic**: Binary label indicating toxic comments.
- **severe_toxic**: Binary label indicating severely toxic comments.
- **obscene**: Binary label indicating obscene comments.
- **threat**: Binary label indicating threats.
- **insult**: Binary label indicating insults.
- **identity_hate**: Binary label indicating identity hate.

## Methodology

### 1. Importing Modules and Loading Data

We start by importing the necessary libraries, including TensorFlow for building the model, Pandas for data manipulation, and NumPy for numerical operations. The dataset is then loaded into a Pandas DataFrame for initial exploration and preprocessing.

### 2. Data Preprocessing

#### Vectorization of Text Data

Text data is converted into a numerical format using TensorFlow's `TextVectorization` layer. This process involves:
- Tokenizing the text data.
- Converting tokens into integer sequences.
- Padding or truncating sequences to a uniform length.

#### Creating TensorFlow Dataset

The vectorized text data and labels are converted into a TensorFlow dataset. The dataset is then shuffled, batched, and prefetched to prepare it for training.

### 3. Building the Sequential Model

We build a Sequential model using Keras with the following layers:
- **Embedding Layer**: Converts integer-encoded words into dense vectors.
- **Bidirectional LSTM Layer**: Processes the sequence in both forward and backward directions to capture context.
- **Dense Layers**: Extract features from the LSTM output with ReLU activation for non-linearity.
- **Output Layer**: Predicts the probabilities of the six types of toxicity using sigmoid activation.

### 4. Model Training

The model is compiled with the Binary Crossentropy loss function and the Adam optimizer. It is trained on the training dataset with validation on the validation set. The trained model is saved for later use.

### 5. Model Evaluation

The model is evaluated using Precision, Recall, and Categorical Accuracy metrics. These metrics provide insights into the model's performance in terms of true positives, false positives, and overall accuracy.

### 6. Making Predictions

We demonstrate how to preprocess input text and use the trained model to predict the probabilities of each type of toxicity for new comments.

### 7. Integrating with Gradio

Gradio is used to create an interactive interface for the model. This interface allows users to input comments and get the predicted probabilities of each type of toxicity.

## Instructions for Running the Notebook

### Prerequisites

Ensure you have the following libraries installed:
- TensorFlow
- Pandas
- NumPy
- Gradio (for the interactive interface)

You can install the required libraries using pip:
```bash
pip install tensorflow pandas numpy gradio
```

### Running the Notebook

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ishans2404/toxic-comment-classification.git
   cd toxic-comment-classification
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook toxic_comment_classification.ipynb
   ```

3. **Run the Cells**:
   Execute the cells sequentially to preprocess the data, build and train the model, and make predictions. 

4. **Launch the Gradio Interface**:
   Run the code in the Gradio section to launch the web interface and interact with the model.

## Conclusion

This project demonstrates a comprehensive approach to building a toxic comment classification model using deep learning techniques. The methodology includes data preprocessing, model building, training, evaluation, and deployment. The interactive Gradio interface provides an accessible way to utilize the model for real-world applications.
\
