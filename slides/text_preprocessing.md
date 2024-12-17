

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_cluster_word_cloud(text):
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    wordcloud = WordCloud().generate(text)

text = '\n'.join(res_df['processed_txt'].values)

plot_cluster_word_cloud(9)
```

# Bag of words

[power of tf-idf](https://habr.com/ru/companies/wildberries/articles/861466/)

simple vectorizer
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    analyzer='word',
    lowercase=True,
    token_pattern=r'\b[\w\d]{3,}\b'
)

vectorizer.fit(df['text'].values)
print('vectorizer fitted')
```

Sophisticated vectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

def tokenize(raw):
    return [w.lower() for w in word_tokenize(raw) if w.isalpha()]

class StemmedTfidfVectorizer(TfidfVectorizer):
    en_stemmer = SnowballStemmer('english')
    
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (StemmedTfidfVectorizer.en_stemmer.stem(w) for w in analyzer(doc))

tfidf = StemmedTfidfVectorizer(
    tokenizer=tokenize, 
    analyzer="word", 
    stop_words='english', 
    ngram_range=(1,1), 
    min_df=40    # limit of minimum number of counts: 3
)
print('Model fit')
tfidf = tfidf.fit(res_df['processed_txt'].values)
```