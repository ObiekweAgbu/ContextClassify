import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import seaborn as sns
import pandas as pd
import tensorflow_hub as hub
import numpy as np

sns.set_style("whitegrid")

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
import re
import tensorflow as tf

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Load hatespeech dataset
df1 = pd.read_csv('../datasets/hate_speech.csv', usecols=[1, 2], header=0, names=['Tweet', 'Label'])
df1 = df1.sample(frac=1).reset_index(drop=True)


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


# label code dictionary
label_codes = {
    'hatespeech': 1,
    'non hate speech': 0
}
# clean dataset
df1 = clean_tweet(df1)

# assign labels
df1['Label_Code'] = df1['Label']
df1 = df1.replace({'Label_Code': label_codes})

print(df1['Label'].head())

# Load Pretrained Word2Vec
embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")


def get_max_length(df):
    """
    get max token counts from train data,
    so we use this number as fixed length input to active learner cell
    """
    max_length = 0
    for row in df['Tweet_Parsed']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))

    print(str(max_length))
    return max_length


def get_word2vec_enc(tweets):
    """
    get word2vec value for each word in sentence.
    concatenate word in numpy array, so we can use it as active learner input
    """
    encoded_tweets = []
    for review in tweets:
        tokens = review.split(" ")
        word2vec_embedding = embed(tokens)
        encoded_tweets.append(word2vec_embedding)
    return encoded_tweets


def get_padded_encoded_reviews(encoded_reviews):
    """
    for short sentences, we prepend zero padding so all input to active learner has same length
    """
    padded_reviews_encoding = []
    for enc_review in encoded_reviews:
        zero_padding_cnt = max_length - enc_review.shape[0]
        pad = np.zeros((1, 250))
        for i in range(zero_padding_cnt):
            enc_review = np.concatenate((pad, enc_review), axis=0)
        padded_reviews_encoding.append(enc_review)
    return padded_reviews_encoding


def preprocess(df):
    """
    encode text value to numeric value
    """
    # encode words into word2vec
    tweets = df['Tweet_Parsed'].tolist()

    encoded_tweets = get_word2vec_enc(tweets)
    padded_encoded_tweets = get_padded_encoded_reviews(encoded_tweets)
    # encoded sentiment
    X = np.array(padded_encoded_tweets)
    return X


# max length is 50
max_length = get_max_length(df1)

# collapse to 2 dimensions for active learner
train_dataset = preprocess(df1)
nsamples, nx, ny = train_dataset.shape
X_raw = train_dataset.reshape((nsamples, nx * ny))
y_raw = np.asarray(df1['Label'].astype('int'))

# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

# train test split for best random forest classifier
X_training, X_test, y_training, y_test = train_test_split(X_train,
                                                          y_train,
                                                          test_size=0.15,
                                                          random_state=8)

# random forest feature training set
with open('../Pickles/X_training.pickle', 'wb') as output:
    pickle.dump(X_training, output)

# random forest label training set
with open('../Pickles/y_training.pickle', 'wb') as output:
    pickle.dump(y_training, output)

# random forest feature test set
with open('../Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)

# random forest label test set
with open('../Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

# Pickled files for the active learner
with open('../Pickles/X_raw.pickle', 'wb') as output:
    pickle.dump(X_raw, output)

with open('../Pickles/y_raw.pickle', 'wb') as output:
    pickle.dump(y_raw, output)

with open('../Pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)

with open('../Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)

with open('../Pickles/X_pool.pickle', 'wb') as output:
    pickle.dump(X_pool, output)

with open('../Pickles/y_pool.pickle', 'wb') as output:
    pickle.dump(y_pool, output)

# df
with open('../Pickles/df1.pickle', 'wb') as output:
    pickle.dump(df1, output)

# reference : https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
# reference : Minsuk Heo, tf2, (2020), GitHub repository,
# reference : P.W.D. Miguel Fernández Zafra,Latest News Classifier, (2019), GitHub repository
