import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Importing the dataset
DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('twitterAnalysis.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

# Removing unnecessary columns
dataset = dataset[['sentiment', 'text']]

# Replacing values to ease understanding
dataset['sentiment'] = dataset['sentiment'].replace(4, 1)

# Plotting the distribution
ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
ax.set_xticklabels(['Negative', 'Positive'], rotation=0)

# Storing data in lists
text, sentiment = list(dataset['text']), list(dataset['sentiment'])

# Emoji dictionary
emojis = {
    ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire',
    ':(': 'sad', ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry',
    ':O': 'surprised', ':-@': 'shocked', ':@': 'shocked',
    ':-$': 'confused', ':\\': 'annoyed', ':#': 'mute', ':X': 'mute',
    ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy', '@@': 'eyeroll',
    ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
    '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
    ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
}

# Stopwords list
stopwordlist = [
    'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'down',
    'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
    'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm',
    'ma', 'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's',
    'same', 'she', 'shes', 'should', 'shouldve', 'so', 'some', 'such', 't', 'than',
    'that', 'thatll', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 've', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'y', 'you',
    'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
]

def preprocess(textdata):
    processedText = []
    wordLemm = WordNetLemmatizer()
    
    # Regex patterns
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(urlPattern, ' URL', tweet)
        
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
            
        tweet = re.sub(userPattern, ' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        
        tweetwords = ''
        for word in tweet.split():
            if word not in stopwordlist and len(word) > 1:
                word = wordLemm.lemmatize(word)
                tweetwords += (word + ' ')
                
        processedText.append(tweetwords)
    
    return processedText

# Preprocessing text
t = time.time()
processedtext = preprocess(text)
print(f'Text Preprocessing complete. Time Taken: {round(time.time()-t)} seconds')

# Generate word clouds
data_neg = processedtext[:800000]
wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(data_neg))
plt.figure(figsize=(20,20))
plt.imshow(wc)

data_pos = processedtext[800000:]
wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(data_pos))
plt.figure(figsize=(20,20))
plt.imshow(wc)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, test_size=0.05, random_state=0)
print('Data Split done.')

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)
print(f'Vectorizer fitted. No. of feature words: {len(vectorizer.get_feature_names_out())}')

X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)
print('Data Transformed')

# Model evaluation function
def model_evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = [f'{value:.2%}' for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='', xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontsize=14)
    plt.ylabel("Actual values", fontsize=14)
    plt.title("Confusion Matrix", fontsize=18)
    plt.show()

# Train and evaluate models
BNBmodel = BernoulliNB(alpha=2)
BNBmodel.fit(X_train, y_train)
model_evaluate(BNBmodel)

SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_evaluate(SVCmodel)

LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=1)
LRmodel.fit(X_train, y_train)
model_evaluate(LRmodel)

# Save models
pickle.dump(vectorizer, open('vectoriser-ngram-(1,2).pickle', 'wb'))
pickle.dump(LRmodel, open('Sentiment-LR.pickle', 'wb'))
pickle.dump(BNBmodel, open('Sentiment-BNB.pickle', 'wb'))

def load_models():
    vectoriser = pickle.load(open('vectoriser-ngram-(1,2).pickle', 'rb'))
    LRmodel = pickle.load(open('Sentiment-LR.pickle', 'rb'))
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    data = []
    for t, pred in zip(text, sentiment):
        data.append((t, pred))
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    return df.replace([0,1], ["Negative", "Positive"])

if __name__ == "__main__":
    vectoriser, LRmodel = load_models()
    text = ["I hate our president", "I Love you.", "Yes! We can win"]
    df = predict(vectoriser, LRmodel, text)
    print(df)
