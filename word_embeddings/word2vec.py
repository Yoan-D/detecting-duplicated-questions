# Train word2vec model using Skip-Gram model
from gensim.models import Word2Vec
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')


def train_word2vec(list1, list2):
    model = Word2Vec(
        list1 + list2,
        size=300,
        min_count=1,
        window=5,
        iter=20,
        sample=1e-4,
        sg=1,  # Skip-Gram model
        hs=1,  # hierarchical softmax
        negative=5)
    # model.wv.init_sims(replace=True) # Precompute L2-normalized vectors.
    return model


def train_word_vectors(cleaned_list_question1, cleaned_list_question2, name):
    word_vectors = train_word2vec(cleaned_list_question1, cleaned_list_question2)
    pickle.dump(word_vectors, open(name + ".p", "wb"))
    print('Done training word2vec model')
    return word_vectors


def get_trained_word_vectors(load_data, cl1, cl2):
    if load_data:
        return pickle.load(open("word_vectors.p", "rb"))
    else:
        return train_word_vectors(cl1, cl2, "word_vectors")


if __name__ == '__main__':
    cleaned_list_question1 = pickle.load(open("deep_cleaned_list_question1.p", "rb"))
    cleaned_list_question2 = pickle.load(open("deep_cleaned_list_question2.p", "rb"))

    w2vec_word_vectors = get_trained_word_vectors(load_data=True, cl1=cleaned_list_question1, cl2=cleaned_list_question2)
