## NLP Project: Text Summarization, Text Classification, and Information Retrieval

### Overview

This project is a comprehensive NLP (Natural Language Processing) pipeline that encompasses three core tasks: Text Summarization, Text Classification, and Information Retrieval. It leverages state-of-the-art deep learning models, such as DistilBERT and T5, and integrates them into a cohesive system to process and analyze large text datasets. The project is structured to allow easy scaling and efficient handling of large datasets, with an emphasis on performance and accuracy.

## Folder Structure

```
nlp_project/
├── Pipfile
├── Pipfile.lock
├── src/
│   ├── data/
│   │   ├── cnn_dailymail/
│   │   │   └── <dataset>.parquet
│   │   ├── wikipedia-22-12-en-embeddings/
│   │   │   └─── <dataset>.parquet
│   │   └── emotion_sentiment_dataset.csv
│   ├── model/
│   │   ├── information-retrieval/
│   │   │   └── distilbert-base-multilingual-cased/
│   │   ├── text-classification/
│   │   │   ├── base/
│   │   │   ├── bi-class/
│   │   │   └── tri-class/
│   │   └── text-summarization/
├── information-retrieval.ipynb
├── information-retrieval.html
├── text-classification.ipynb
├── text-classification.html
├── text-summarization.ipynb
├── text-summarization.html
└── README.md
```

## Project Components

1. **Text Summarization**

	*	Task: Automatically generate concise and coherent summaries of long texts.
	*	Model: The summarization model is based on the `T5` transformer architecture, which is known for its efficiency and performance in text generation tasks.
	*	Data: Uses the **CNN/DailyMail** dataset, which contains news articles paired with human-generated summaries.
	*	Scripts/Notebooks - `text-summarization.ipynb`: Jupyter notebook containing the code for training and evaluating the summarization model.

2. **Text Classification**

	*	Task: Classify text into predefined categories, such as sentiment analysis or topic categorization.
	*	Model: The classification model is built using the `BERT` architecture (Bi-class, Tri-class models), which is fine-tuned for specific classification tasks.
	*	Data: Uses a sentiment analysis dataset (e.g., **Emotion Sentiment Dataset**).
	*	Scripts/Notebooks - `text-classification.ipynb`: Jupyter notebook containing the code for training, evaluating, and fine-tuning the classification models.

3. **Information Retrieval**

	*	Task: Retrieve the most relevant document from a large dataset based on a given query.
	*	Model: Uses a pre-trained `DistilBERT` model to compute embeddings for documents and queries, and performs similarity search using cosine similarity.
	*	Data: Uses the Wikipedia dataset with precomputed embeddings for fast retrieval.
	*	Scripts/Notebooks - `information-retrieval.ipynb`: Jupyter notebook demonstrating how to perform information retrieval using the precomputed embeddings.

## Data

1. CNN/DailyMail Dataset

* Located in src/data/cnn_dailymail/, this dataset is used for training the summarization model.

2. Emotion Sentiment Dataset

* Stored as emotion_sentiment_dataset.csv in src/data/, this dataset is used for training the sentiment analysis model.

3. Wikipedia Embeddings

* Precomputed embeddings from the Wikipedia dataset are stored in src/data/wikipedia-22-12-en-embeddings/.

## Model Checkpoints

Model checkpoints are stored in the model/ directory, organized by task:

	•	information-retrieval/: Contains the DistilBERT model for information retrieval.
	•	text-classification/: Contains BERT models for various classification tasks.
	•	text-summarization/: Contains the T5 model for text summarization.

## Results

Each task outputs results in the form of metrics, generated text, or retrieved documents. Check the respective notebook HTML for details on how to interpret and visualize the results.

### Future Work

	•	Optimization: Improve the performance of models through hyperparameter tuning and model pruning.
	•	Deployment: Package the models into a web service for real-time inference.
	•	Integration: Combine the summarization and classification models into a pipeline for more complex NLP tasks.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

	•	Hugging Face: For providing the pre-trained models and the transformers library.
	•	CNN/DailyMail: For the dataset used in text summarization.
	•	Wikipedia: For the data used in information retrieval.