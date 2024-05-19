import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models

# Load data
def load_data(filepath, column_name='Response 4'):
    data = pd.read_excel(filepath)
    return data[column_name].dropna()

# Preprocess data
def preprocess_data(texts):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_texts = []
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        preprocessed_texts.append(cleaned_tokens)
    return preprocessed_texts

# Build LDA model
def build_lda_model(texts, num_topics=10):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary

# Visualize topics
def visualize_topics(lda_model, corpus, dictionary):
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, 'lda.html')


def main():
    filepath = 'Lab7Responses.xlsx'
    responses = load_data(filepath)
    preprocessed_texts = preprocess_data(responses)
    lda_model, corpus, dictionary = build_lda_model(preprocessed_texts)
    visualize_topics(lda_model, corpus, dictionary)
    print(lda_model.print_topics(num_words=22))

if __name__ == "__main__":
    main()
