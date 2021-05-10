import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import matplotlib as mpl
import matplotlib.pyplot as plt
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

path_df = "Pickles/df1.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# features_train
path_features_train = "Pickles/X_raw.pickle"
with open(path_features_train, 'rb') as data:
    X_raw = pickle.load(data)

# labels_train
path_labels_train = "Pickles/y_raw.pickle"
with open(path_labels_train, 'rb') as data:
    y_raw = pickle.load(data)

# features_test
path_features_test = "Pickles/X_train.pickle"
with open(path_features_test, 'rb') as data:
    X_train = pickle.load(data)

# labels_test
path_labels_test = "Pickles/y_train.pickle"
with open(path_labels_test, 'rb') as data:
    y_train = pickle.load(data)

# labels_test
path_x_pool = "Pickles/X_pool.pickle"
with open(path_x_pool, 'rb') as data:
    X_pool = pickle.load(data)

path_label_pool = "Pickles/y_pool.pickle"
with open(path_label_pool, 'rb') as data:
    y_pool = pickle.load(data)

path_label_pool = "Pickles/rfc_hatespeech.pickle"
with open(path_label_pool, 'rb') as data:
    best_rfc = pickle.load(data)

pca = PCA(n_components=2, random_state=8)
transformed_tweets = pca.fit_transform(X=X_raw)

# Isolate the data we'll need for plotting.
x_component, y_component = transformed_tweets[:, 0], transformed_tweets[:, 1]

# Plot our dimensionality-reduced (via PCA) dataset.
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8 / 10, label=str(y_component))
plt.legend(loc='lower right')
plt.title('Tweet classes after PCA transformation')
plt.show()

# initializing the learner
learner = ActiveLearner(
    estimator=best_rfc,
    X_training=X_train, y_training=y_train
)

# ...obtaining new labels from the Oracle...

# supply label for queried instance
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)

print(predictions)

unqueried_score = learner.score(X_raw, y_raw)
print(unqueried_score)

# Plot our classification results.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.scatter(x=x_component[is_correct], y=y_component[is_correct], c='g', marker='+', label='Correct', alpha=8 / 10)
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8 / 10)
ax.legend(loc='lower right')
ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
plt.show()

# total number of queries to be done on model
N_QUERIES = 1000

performance_history = [unqueried_score]


def convert(list):
    # Converting integer list to string list
    s = [str(i) for i in list]

    # Join list items using join()
    res = int("".join(s))

    return (res)


# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).

for index in range(N_QUERIES):
    query_index, query_instance = learner.query(X_pool)

    # Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool.
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    # Calculate and report our model's accuracy.
    model_accuracy = learner.score(X_raw, y_raw)
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

    # Save our model's performance for plotting.
    performance_history.append(model_accuracy)

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

plt.show()

# Isolate the data we'll need for plotting.
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)

# Plot our updated classification results once we've trained our learner.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.scatter(x=x_component[is_correct], y=y_component[is_correct], c='g', marker='+', label='Correct', alpha=8 / 10)
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8 / 10)

ax.set_title(
    'Classification accuracy after {n} queries: {final_acc:.3f}'.format(n=N_QUERIES, final_acc=performance_history[-1]))
ax.legend(loc='lower right')

plt.show()

with open('Pickles/active_learner_hate.pickle', 'wb') as output:
    pickle.dump(learner, output)

# reference: https://modal-python.readthedocs.io/en/latest/content/examples/pool-based_sampling.html
