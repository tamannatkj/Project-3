# Amazon Product Review Sentiment Analysis

## Project Overview
This project implements a deep learning-based sentiment classification system for Amazon product reviews. The goal is to classify reviews as either **POSITIVE** or **NEGATIVE**.

To analyze performance and interpretability, this project implements and compares two distinct model architectures:
1.  **LSTM (Long Short-Term Memory):** A LSTM trained from scratch using word embeddings to capture sequential dependencies.
2.  **Transformer (DistilBERT):** A pre-trained BERT-based model fine-tuned on the dataset, utilizing self-attention mechanisms for superior context understanding.

Key features include:
* **Data Preprocessing:** Cleaning, tokenization, and handling class imbalance using weighted loss functions.
* **Model Comparison:** Side-by-side evaluation of LSTM and Transformer performance.
* **Interpretability:** Visualization of Attention Heatmaps to understand which words influence the Transformer's predictions.

## Setup Instructions

### 1. Prerequisites
* Python 3.8 or higher
* Git

### 2. Installation
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone <your-repo-url>
cd amazon-sentiment-analysis

# Create a virtual environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation
The datasets used in this project are publicly available on Kaggle. Please download them from the link below:

Dataset Link: **https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products**, 
- `1429_1.csv`
- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`
- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`

Please create a folder named `data/` in the project root, extract and place the following three CSV files into the data/ folder


## How to Run
**Option 1: Run the Demo (Inference & Visualization)**

To quickly test the models and see attention maps without training, run the Jupyter Notebook demo:

1. Navigate to the demo folder:
```bash
cd demo
```
2. Start Jupyter Notebook:
```bash
jupyter notebook demo.ipynb
```
3. Run all cells. The notebook will:
- Load the pre-trained models.
- Predict sentiment for sample reviews.
- Generate and save an attention heatmap.

**Option 2: Train the Models**

To retrain both models from scratch:

```bash
python src/main.py
```
This script will:

- Load and preprocess the data.
- Train the LSTM model and save the best version.
- Fine-tune the DistilBERT Transformer model.
- Evaluate both models on the test set.
- Save the trained models to models/ and evaluation metrics to results/.

## Expected Output

### Demo Output
After running `demo.ipynb`, you will see:

1.**Prediction Table**: A comparison of predictions from both models on sample text. 

| Review                                           | LSTM      | Transformer |
|-------------------------------------------------|----------|-------------|
| This product is absolutely amazing! I use it... | POSITIVE | POSITIVE    |
| Terrible quality. It broke after one use...     | NEGATIVE | NEGATIVE    |

2. **Attention Heatmap**: A visualization image (**results/attention_heatmap.png**) showing how the Transformer focuses on key sentiment words (e.g., "amazing", "terrible").
3. **Metrics**: A summary of the models' performance on the validation set.

### Training Output
Upon completion of `src/main.py`, a comparison table will be generated in **results/model_comparison.csv**:
 
| Model       | Accuracy | F1-Macro |
|------------|---------|----------|
| LSTM       | 0.9388  | 0.7851   |
| Transformer| 0.9816  | 0.9129   |


## Pre-trained Model Link
To run the demo without retraining the models, please download the pre-trained model artifacts (both LSTM and Transformer) from the link below:

**https://drive.google.com/drive/folders/1aq08vMzVbhGcxJe1DG1ndvmdZvXLknqc?usp=drive_link**

**Setup Instructions:**
1.  Download the ZIP file from the link above.
2.  Extract the contents into the `models/` directory in the root of this repository.
3.  Ensure your directory structure looks like this:
    ```text
    models/
    ├── final_lstm_model/
    │   ├── model.pt
    │   ├── config.json
    │   └── vocab.json
    └── final_transformer_model/
        ├── config.json
        ├── pytorch_model.bin (or model.safetensors)
        ├── vocab.txt
        └── ...
    ```

## Acknowledgments
This project references the following resources:

- Data Preprocessing Strategy: Adapted from the Kaggle notebook by kfrawee.
  Source: [Sentimental Analysis on Kaggle](https://www.kaggle.com/code/kfrawee/sentimental-analysis)

- LSTM Model Architecture: Inspired by the Amazon Sentiment analysis project by iamirmasoud.
  Source: [Amazon Sentiment GitHub Repository](https://github.com/iamirmasoud/amazon_sentiment/tree/main)

- Transformer Model: Utilizes DistilBERT from Hugging Face Transformers.
  Dataset: [Consumer Reviews of Amazon Products provided by Datafiniti.](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)


