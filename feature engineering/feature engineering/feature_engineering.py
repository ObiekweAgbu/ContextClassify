import tweepy
from textblob import TextBlob
import numpy as np
import keys
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from tweepy import OAuthHandler, Stream, StreamListener
import pandas as pd
import pickle
import seaborn as sns
import random
sns.set_style("whitegrid")
import json
import os

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

# assigning of key tokens and keys form hidden environmental variables
key = os.getenv('API_KEY')
secret_key = os.getenv('API_SECRET_KEY')
token = os.getenv('API_TOKEN')
secret_token = os.getenv('API_SECRET_TOKEN')

# initialising of API authetication object
authenticate = tweepy.OAuthHandler(key, secret_key)
authenticate.set_access_token(token, secret_token)
api = tweepy.API(authenticate, wait_on_rate_limit=True)

# Extraction of dataset and loading into pandas dataframe
search_words = ['#BLM', '#COVID19', '#BorisJohnson', '#vaccination', '#Bitcoin']
date_since = "2020-05-20"

posts = tweepy.Cursor(api.search,
                      q=search_words[1],
                      lang="en",
                      since=date_since).items(8000)
df1 = pd.DataFrame([tweet.text for tweet in posts], columns=['Tweet'])
print(df1.head())

# Load data from csv into pandas dataframe
# df = pd.read_csv('training.1600000.processed.noemoticon.csv', usecols = [5], header=None, names = ['Tweet'])
# df1 = pd.read_csv('Sentiment Analysis Dataset.csv', usecols = [1, 3], header = 0, names = ['Label', 'Tweet'])
# df.index = np.array(list(range(0,1600000)))
# df1 = df1.sample(frac=1).reset_index(drop=True)
# df1 = df1.truncate(before=0, after=10000)
df1['Label'] = 'no_label'


# method to prepare tweet dataframe for processing
def clean_tweet(df):
    # removing new lines and correcting other formatting errors
    df['parse1'] = df['Tweet'].str.replace("\r", " ")
    df['parse2'] = df['parse1'].str.replace("\n", " ")
    df['parse3'] = df['parse2'].str.replace("    ", " ")
    df['parse4'] = df['parse3'].str.replace("\b", " ")

    # changing to lowercase so meaning is the same capitalised or not
    df['parse5'] = df['parse4'].str.lower()

    # replace posessive pronouns such as 's
    df['parse6'] = df['parse5'].str.replace("'s", "")

    # removal of punction marks, interchangeble function
    punctuation_signs = list("?:!.,;")
    df['parse7'] = df['parse6']

    for punct_sign in punctuation_signs:
        df['parse8'] = df['parse7'].str.replace(punct_sign, ' ' + punct_sign + ' ')

    nltk.download('punkt')
    print("------------------------------------------------------------")
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []

    for row in range(0, nrows):
        df.at[row, 'parse8'] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', df.at[row, 'parse8'],
                                      flags=re.MULTILINE)  # removal of hyperlinks
        df.at[row, 'parse8'] = re.sub('@[^\s]+', '', df.at[row, 'parse8'])  # removal of @ and mentions
        df.at[row, 'parse8'] = re.sub(r'#', '', df.at[row, 'parse8'])  # removal of hashtags and #
        df.at[row, 'parse8'] = re.sub(r'rt[\s]+', '', df.at[row, 'parse8'])  # removal of  RT's and quote tweets
        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = df.loc[row]['parse8']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    df['parse8'] = lemmatized_text_list

    # Cleaning for stop words
    nltk.download('stopwords')

    # Loading the stop words in english
    stop_words = list(stopwords.words('english'))

    # loop through stop words
    df['parse9'] = df['parse8']

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['parse9'] = df['parse9'].str.replace(regex_stopword, '')

    list_columns = ["Tweet", "parse9", "Label"]
    df = df[list_columns]
    df = df.rename(columns={'parse9': 'Tweet_Parsed'})
    return df


# method to remove redunndant rows and fields
def remove_redundancy(df1):
    for row in range(0, len(df1)):
        text = str(df1.at[row, 'Label'])
        splitter = '"'
        take = text.split(splitter)
        for i in range(0, len(take)):
            if take[i] == "value":
                df1.at[row, 'Label'] = str(take[i + 2].lower())

    for row in range(0, len(df1)):
        if str(df1.at[row, 'Label']) == '{}' or str(df1.at[row, 'Label']) == 'hate_speech':
            df1 = df1.drop([row])

    return df1


# assign subjectivity score to tweet text
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# assign polarity score to tweet text
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# assign label based on polarity score
def getAnalysis(score):
    if score < 0:
        if score < -0.5:
            return 'very negative'
        else:
            return 'negative'
    elif score == 0:
        return 'neutral'
    elif score > 0:
        if score > 0.5:
            return 'very positive'
        else:
            return 'positive'


# label code dictionary
label_codes = {
    'very positive': 4,
    'positve': 3,
    'neutral': 2,
    'negative': 1,
    'very negative': 0
}

df1 = clean_tweet(df1)
df1 = remove_redundancy(df1)

# Label assign process
df1['Subjectivity'] = df1['Tweet_Parsed'].apply(getSubjectivity)
df1['Polarity'] = df1['Tweet_Parsed'].apply(getPolarity)
df1['Label'] = df1['Polarity'].apply(getAnalysis)

df1['Label_Code'] = df1['Label']

# label code assign
df1 = df1.replace({'Label_Code': label_codes})

print(df1['Label'].head())

# Training and test set split
X_train, X_test, y_train, y_test = train_test_split(df1['Tweet_Parsed'],
                                                    df1['Label_Code'],
                                                    test_size=0.15,
                                                    random_state=8)
# tf idf settings
ngram_range = (1, 2)
min_df = 1
max_df = 1.
max_features = 10

# label code format for training and test set
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# tf idf initialisation
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

# training and test set pre processing
features_train = tfidf.fit_transform(X_train).toarray()
features_pool = tfidf.fit_transform(df1['Tweet_Parsed']).toarray()
labels_train = y_train
print(features_train.shape)

# features_test = word2vec.Word2Vec(X_test, workers=4 , min_count=40,  window=5, sample=1e-3)
features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

for Sentiment, category_id in sorted(label_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Sentiment))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

# X_test
with open('Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)

# y_test
with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

# df
with open('../Pickles/df1.pickle', 'wb') as output:
    pickle.dump(df1, output)

# features_train
with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)

# TF-IDF object
with open('../Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)

print(df1.loc[1, ['Label']])

# reference : P.W.D. Miguel Fernández Zafra,Latest News Classifier, (2019), GitHub repository
