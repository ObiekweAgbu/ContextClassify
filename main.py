"""
University of the West of England Bristol
Author: Obiekwe Joseph Agbu
Student Number: 18037612
Date : 20 November 2020
Course : Digital Systems Project
Course Code : UFCFXK-30-3
Tutor : Dr. Chris Simons
"""
import tweepy
import numpy as np
from tkinter import *
import pickle
import keys
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import os
import tensorflow_hub as hub
from textblob import TextBlob

path_df = "Pickles/active_learner.pickle"
with open(path_df, 'rb') as data:
    learner = pickle.load(data)

path_df = "Pickles/active_learner_hate.pickle"
with open(path_df, 'rb') as data:
    hate_learner = pickle.load(data)

# assigning of key tokens and keys form hidden environmental variables
key = os.getenv('API_KEY')
secret_key = os.getenv('API_SECRET_KEY')
token = os.getenv('API_TOKEN')
secret_token = os.getenv('API_SECRET_TOKEN')

# initialising of API authetication object
authenticate = tweepy.OAuthHandler(key, secret_key)
authenticate.set_access_token(token, secret_token)
api = tweepy.API(authenticate, wait_on_rate_limit=True)

embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")


def show_last():
    try:
        status.configure(text="Enter handle or hashtag beginning with @ or #")
        term = str(twitter_term.get())

        # if entered term is a user handle
        if term[0] == '@':
            posts = api.user_timeline(screen_name=term[1:], count=5, lang="en", tweet_mode="extended")
            tweet = str(posts[0].full_text)

        # if entered term is a hashtag
        if term[0] == '#':
            posts = api.search(q=term,
                               lang="en", count=1,
                               result_type="recent")
            tweet = str(posts[0].text)

        # enable all text boxes and clear text
        sentiment_text.configure(state='normal')
        sentiment_text.delete("1.0", "end")
        tweet_display.configure(state='normal')
        tweet_display.delete("1.0", "end")
        hate_status.configure(state='normal')
        hate_status.delete("1.0", "end")
        entities.configure(state='normal')
        entities.delete("1.0", "end")
        # display tweet
        tweet_display.insert(INSERT, tweet)
        tweet = clean(tweet)

        # display context data
        sentiment_text.insert(INSERT, str(get_sentiment(tweet)))
        hate_status.insert(INSERT, str(get_hatespeech(tweet)))
        entities.insert(INSERT, str(return_entity(tweet)))


    except:
        # invalid input error condition
        print("An exception occurred")
        tweet_display.delete("1.0", "end")
        status.configure(text="Enter a VALID handle or hashtag e.g @POTUS, #covid")


# method to return named entities
def return_entity(input):
    input1 = TextBlob(input)
    # grammar correct cleaned tweet
    text = str(input1.correct())

    # tokenise and chunk text based on part of speech
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(pos_tags, binary=False)  # either NE or not NE
    for chunk in chunks:
        print(chunk)

    entities = []
    labels = []

    # if chunk has label attached add label and entity to designated list
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            print(chunk)
            entities.append(' '.join(c[0] for c in chunk))
            labels.append(chunk.label())

    # create dictionary with both lists
    keys_list = entities
    values_list = labels
    zip_iterator = zip(keys_list, values_list)

    # return dictionary in string form
    a_dictionary = dict(zip_iterator)
    result = str(a_dictionary)

    return result


def preprocess(tweet, max_length):
    """
    encode text value to numeric value
    """
    # encode words into word2vec
    encoded_tweet = word_vec_tweet(tweet)
    padded_encoded_tweet = padding_word2vec(encoded_tweet, max_length)
    # encoded sentiment
    X = np.array(padded_encoded_tweet)
    return X


# modified clean tweet method to clean single tweet extracted from twitter
def clean(tweet):
    lemmatizer = WordNetLemmatizer()
    cleaned_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+'')", " ", tweet).split())
    cleaned_tweet = cleaned_tweet.replace("\r", " ")
    cleaned_tweet = cleaned_tweet.replace("\n", " ")
    cleaned_tweet = cleaned_tweet.replace("    ", " ")
    cleaned_tweet = cleaned_tweet.replace("\b", " ")
    cleaned_tweet = cleaned_tweet.lower()
    cleaned_tweet = cleaned_tweet.replace("'s", "")
    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        cleaned_tweet = cleaned_tweet.replace(punct_sign, ' ' + punct_sign + ' ')

    # nltk.download('punkt')
    # print("------------------------------------------------------------")
    # nltk.download('wordnet')

    cleaned_tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', cleaned_tweet,
                          flags=re.MULTILINE)  # removal of hyperlinks
    cleaned_tweet = re.sub('@[^\s]+', '', cleaned_tweet)  # removal of @ and mentions
    cleaned_tweet = re.sub(r'#', '', cleaned_tweet)  # removal of hashtags and #
    cleaned_tweet = re.sub(r'rt[\s]+', '', cleaned_tweet)  # removal of  RT's and quote tweets


    text_words = cleaned_tweet.split(" ")

    lemmatized_list = []
    for word in text_words:
        lemmatized_list.append(lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)
    tweet = lemmatized_text
    # Cleaning for stop words
    # nltk.download('stopwords')

    # Loading the stop words in english
    stop_words = list(stopwords.words('english'))
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        tweet = tweet.replace(regex_stopword, '')

    return tweet


def padding_word2vec(word2vec_embedding, max_length):
    """
      padding so all input to active learner has same length as data used to train
        """
    zero_padding_cnt = max_length - word2vec_embedding.shape[0]
    pad = np.zeros((1, 250))
    for i in range(zero_padding_cnt):
        word2vec_embedding = np.concatenate((pad, word2vec_embedding), axis=0)
    return word2vec_embedding


def word_vec_tweet(tweet):
    """
        get word2vec value for each word in sentence.
        concatenate word in numpy array for prediction
        """
    tokens = tweet.split(" ")
    if len(tokens) > 37:
        tokens = tokens[0:37]
    word2vec_embedding = embed(tokens)
    return word2vec_embedding


# retrieve sentiment
def get_sentiment(tweet):
    train_data = preprocess(tweet, 37)
    nx, ny = train_data.shape
    vector = train_data.reshape(nx * ny)
    sent = learner.predict(vector.reshape(1, -1))

    # convert label code to string sentiment value
    if str(sent) == '[0]':
        sent = 'Hate Speech'
    elif str(sent) == '[1]':
        sent = 'Very Negative'
    elif str(sent) == '[2]':
        sent = 'Negative'
    elif str(sent) == '[3]':
        sent = 'Neutral'
    elif str(sent) == '[4]':
        sent = 'Positive'
    elif str(sent) == '[5]':
        sent = 'Very Positive'

    return sent


# retrieve hate speech status
def get_hatespeech(tweet):
    train_data = preprocess(tweet, 50)
    nx, ny = train_data.shape
    vector = train_data.reshape(nx * ny)
    sent = hate_learner.predict(vector.reshape(1, -1))

    # convert label code to string hate speech status
    if str(sent) == '[0]':
        sent = 'False'
    elif str(sent) == '[1]':
        sent = 'True'
    return sent


# root action pane
root = Tk()
root.geometry("500x500")
root.iconbitmap('C:\Context_Classify\context-classify-logo.ico')
root.title('ContextClassify')

clicked = StringVar()
# drop_select = OptionMenu(root, clicked, "Test 1", "Test 2")
# drop_select.grid(row=0, column=0)

# Term entry box
twitter_term = Entry(root, width=35, borderwidth=5)
twitter_term.grid(row=2, column=0)

printTweet = Button(root, text="Print Last Tweet", command=show_last)
printTweet.grid(row=2, column=1)

# Tweet display Box
tweet_display = Text(root, height=10, width=30)
tweet_display.grid(row=5, column=1)
tweet_display.configure(state='disable')
display_label = Label(root, text="Tweet:")
display_label.grid(row=5, column=0)

# Sentiment Display Box
sentiment_text = Text(root, height=2, width=30)
sentiment_text.grid(row=10, column=1)
sentiment_text.configure(state='disable')
sentiment_label = Label(root, text="Sentiment:")
sentiment_label.grid(row=10, column=0)

# Entity Display Box
entities = Text(root, height=5, width=30)
entities.grid(row=15, column=1)
entities.configure(state='disable')
entity_label = Label(root, text="Entities:")
entity_label.grid(row=15, column=0)

# Hate Speech  Status Display Box
hate_status = Text(root, height=5, width=30)
hate_status.grid(row=20, column=1)
hate_status.configure(state='disable')
hate_label = Label(root, text="Hate Speech:")
hate_label.grid(row=20, column=0)

# status bar
status = Label(root, text="Enter handle or hashtag beginning with @ or #")
status.grid(row=1, column=0)

root.mainloop()

# reference : raaga500 ,YTshared, (2020), GitHub repository