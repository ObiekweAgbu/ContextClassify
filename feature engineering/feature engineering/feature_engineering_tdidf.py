import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd
import pickle
import seaborn as sns

sns.set_style("whitegrid")
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data from csv into pandas dataframe
df1 = pd.read_csv('../datasets/Sentiment Analysis Dataset.csv', usecols=[1, 3], header=0, names=['Label', 'Tweet'])
df1 = df1.sample(frac=1).reset_index(drop=True)
df1 = df1.truncate(before=0, after=10000)


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
    'positive': 1,
    'negative': 0
}

df1 = clean_tweet(df1)

df1['Label_Code'] = df1['Label']

print(df1['Label'].head())

# tf idf settings
ngram_range = (1, 2)
min_df = 1
max_df = 1.
max_features = 10

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


X_raw = tfidf.fit_transform(df1['Tweet_Parsed']).toarray()
y_raw = np.asarray(df1['Label_Code'].astype('int'))

# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples + 1, size=3)

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)

# Training and test set split
X_training, X_test, y_training, y_test = train_test_split(X_train,
                                                          y_train,
                                                          test_size=0.15,
                                                          random_state=8)

# best random forest feature training data
with open('../Pickles/X_training.pickle', 'wb') as output:
    pickle.dump(X_training, output)

# best random forest label training data
with open('../Pickles/y_training.pickle', 'wb') as output:
    pickle.dump(y_training, output)

# best random forest feature test data
with open('../Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)

# best random forest label test data
with open('../Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

#####################################################################################################################################################################################################
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

# reference : P.W.D. Miguel Fernández Zafra,Latest News Classifier, (2019), GitHub repository