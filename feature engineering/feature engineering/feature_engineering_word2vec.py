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

# load sentiment labelled dataset into dataframe
df1 = pd.read_csv('../datasets/2500LabelledSentiment.csv', usecols=[2, 3], header=0, names=['Tweet', 'Label'])


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


# method to remove redundant rows and fields
def remove_redundancy(df1):
    for row in range(0, len(df1)):
        text = str(df1.at[row, 'Label'])
        splitter = '"'
        take = text.split(splitter)
        for i in range(0, len(take)):
            if take[i] == "value":
                df1.at[row, 'Label'] = str(take[i + 2].lower())

    for row in range(0, len(df1)):
        if str(df1.at[row, 'Label']) == '{}':
            df1 = df1.drop([row])

    return df1


# label code dictionary
label_codes = {
    'very_positive': 5,
    'positive': 4,
    'neutral': 3,
    'negative': 2,
    'very_negative': 1,
    'hate_speech': 0
}

df1 = clean_tweet(df1)
df1 = remove_redundancy(df1)

df1['Label_Code'] = df1['Label']
df1 = df1.replace({'Label_Code': label_codes})

print(df1['Label'].head())

"""
   Self trained word2vec encoder
    """

# # Generates skip-gram pairs with negative sampling for a list of sequences
# # (int-encoded sentences) based on window size, number of negative samples
# # and vocabulary size.
# def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
#     # Elements of each training example are appended to these lists.
#     targets, contexts, labels = [], [], []
#
#     # Build the sampling table for vocab_size tokens.
#     sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
#
#     # Iterate over all sequences (sentences) in dataset.
#     for sequence in tqdm.tqdm(sequences):
#
#         # Generate positive skip-gram pairs for a sequence (sentence).
#         positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
#             sequence,
#             vocabulary_size=vocab_size,
#             sampling_table=sampling_table,
#             window_size=window_size,
#             negative_samples=0)
#
#         # Iterate over each positive skip-gram pair to produce training examples
#         # with positive context word and negative samples.
#         for target_word, context_word in positive_skip_grams:
#             context_class = tf.expand_dims(
#                 tf.constant([context_word], dtype="int64", shape=None), 1)
#             negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
#                 true_classes=context_class,
#                 num_true=1,
#                 num_sampled=num_ns,
#                 unique=True,
#                 range_max=vocab_size,
#                 seed=SEED,
#                 name="negative_sampling")
#
#             # Build context and label vectors (for one target word)
#             negative_sampling_candidates = tf.expand_dims(
#                 negative_sampling_candidates, 1)
#
#             context = tf.concat([context_class, negative_sampling_candidates], 0)
#             label = tf.constant([1] + [0] * num_ns, dtype="int64")
#
#             # Append each element from the training example to global lists.
#             targets.append(target_word)
#             contexts.append(context)
#             labels.append(label)
#
#     return targets, contexts, labels
#
#
# # Define the vocabulary size and number of words in a sequence.
# vocab_size = 4096
# sequence_length = 10
#
#
# # a custom standardization function to lowercase the text and
# # remove punctuation.
# def custom_standardization(input_data):
#     lowercase = tf.strings.lower(input_data)
#     return tf.strings.regex_replace(lowercase,
#                                     '[%s]' % re.escape(string.punctuation), '')
#
#
# # Define the vocabulary size and number of words in a sequence.
# vocab_size = 13277
# sequence_length = 10
#
# tweet_array = df1['Tweet_Parsed'].to_numpy()
# tweet_ds = tf.data.Dataset.from_tensor_slices((tweet_array))
# # Use the text vectorization layer to normalize, split, and map strings to
# # integers. Set output_sequence_length length to pad all samples to same length.
# vectorize_layer = TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=vocab_size,
#     output_mode='int',
#     output_sequence_length=sequence_length)
#
# tweet_ds = tweet_ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))
#
# vectorize_layer.adapt(tweet_ds.batch(1024))
#
# inverse_vocab = vectorize_layer.get_vocabulary()
# print(inverse_vocab[:20])
# # Vectorize the data in text_ds.
# text_vector_ds = tweet_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
#
# sequences = list(text_vector_ds.as_numpy_iterator())
# print(len(sequences))
#
# for seq in sequences[:5]:
#     print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
#
# targets, contexts, labels = generate_training_data(
#     sequences=sequences,
#     window_size=2,
#     num_ns=4,
#     vocab_size=vocab_size,
#     seed=SEED)
#
# print(len(targets), len(contexts), len(labels))
#
# BATCH_SIZE = 1024
# BUFFER_SIZE = 10000
# dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# print(dataset)
#
# dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
# print(dataset)
#
#
# class Word2Vec(Model):
#     def __init__(self, vocab_size, embedding_dim):
#         super(Word2Vec, self).__init__()
#         self.target_embedding = Embedding(vocab_size,
#                                           embedding_dim,
#                                           input_length=1,
#                                           name="w2v_embedding")
#         self.context_embedding = Embedding(vocab_size,
#                                            embedding_dim,
#                                            input_length=4 + 1)
#         self.dots = Dot(axes=(3, 2))
#         self.flatten = Flatten()
#
#     def call(self, pair):
#         target, context = pair
#         word_emb = self.target_embedding(target)
#         context_emb = self.context_embedding(context)
#         dots = self.dots([context_emb, word_emb])
#         return self.flatten(dots)
#
#     def custom_loss(x_logit, y_true):
#         return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
#
#
# embedding_dim = 1
# word2vec = Word2Vec(vocab_size, embedding_dim)
# word2vec.compile(optimizer='adam',
#                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                  metrics=['accuracy'])
#
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
#
# word2vec.fit(dataset, epochs=50, callbacks=[tensorboard_callback])
#
# weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
# vocab = vectorize_layer.get_vocabulary()
# dict = {vocab[i]: weights[i] for i in range(len(vocab))}
# df = pd.DataFrame(inverse_vocab, columns=['vocab'])
# df['vocab'] = df['vocab'].replace(dict, regex=True)
# word2vec[inverse_vocab]

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


max_length = get_max_length(df1)


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


def get_padded_encoded_tweets(encoded_reviews):
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
    padded_encoded_tweets = get_padded_encoded_tweets(encoded_tweets)
    # encoded sentiment
    X = np.array(padded_encoded_tweets)
    return X


# reshape array to fit active learner estimator
train_dataset = preprocess(df1)
nsamples, nx, ny = train_dataset.shape
X_raw = train_dataset.reshape((nsamples, nx * ny))

y_raw = np.asarray(df1['Label_Code'].astype('int'))

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
