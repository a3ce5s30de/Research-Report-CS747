import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from transformers import pipeline

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess data
def load_data(filepath):
    data = pd.read_excel(filepath)
    response_data = data['Response 3'].dropna()
    return response_data

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words and word.isalpha()])

# Initialize BERT sentiment analysis pipeline only once
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Sentiment Analysis using BERT in a batch processing manner
def sentiment_analysis(texts):
    # BERT can handle batch predictions, which is more efficient
    results = classifier(texts)
    return [(result['label'], result['score']) for result in results]

# Topic Modeling
def topic_modeling(data, n_components=5, no_top_words=33):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    data_vectorized = vectorizer.fit_transform(data)
    lda_model = LatentDirichletAllocation(n_components=n_components, random_state=0)
    lda_model.fit(data_vectorized)
    feature_names = vectorizer.get_feature_names_out()
    topics = {f"Topic {i}": [feature_names[index] for index in topic.argsort()[:-no_top_words - 1:-1]] for i, topic in enumerate(lda_model.components_)}
    return topics

# Plot sentiment distribution
def plot_sentiment_distribution(sentiment_scores):
    plt.figure(figsize=(8, 6))
    plt.hist([score for _, score in sentiment_scores], bins=30, color='blue', alpha=0.7)
    plt.title('Sentiment Distribution of All Responses')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def main():
    filepath = 'Lab7Responses.xlsx'
    response_data = load_data(filepath)
    texts = [preprocess_text(text) for text in response_data]  # Preprocess texts for sentiment analysis

    # Sentiment Analysis
    sentiment_results = sentiment_analysis(texts)

    # Plotting sentiment distribution
    plot_sentiment_distribution(sentiment_results)

    # Separate positive and negative sentiments for topic modeling
    positive_responses = [text for text, (label, _) in zip(texts, sentiment_results) if label == 'POSITIVE']
    negative_responses = [text for text, (label, _) in zip(texts, sentiment_results) if label == 'NEGATIVE']

    # Topic Modeling on positive and negative responses
    positive_topics = topic_modeling(positive_responses)
    negative_topics = topic_modeling(negative_responses)

    print("Positive Sentiment Topics:")
    for topic, words in positive_topics.items():
        print(f"{topic}: {', '.join(words)}")

    print("\nNegative Sentiment Topics:")
    for topic, words in negative_topics.items():
        print(f"{topic}: {', '.join(words)}")

if __name__ == "__main__":
    main()
