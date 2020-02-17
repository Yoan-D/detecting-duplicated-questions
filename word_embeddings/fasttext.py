# Train fasttext model
from gensim.models import FastText
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')


def train_fasttext(list1, list2):
    model = FastText(
        list1 + list2,
        size=300,
        min_count=2,
        window=5,
        iter=20,
        sample=1e-4,
        negative=10)
    model.wv.init_sims(replace=True)  # normalize word vectors
    return model


def train_fasttext_vectors(cleaned_list_question1, cleaned_list_question2, name):
    word_vectors = train_fasttext(cleaned_list_question1, cleaned_list_question2)
    pickle.dump(word_vectors, open(name + ".p", "wb"))
    print('Done training fasttext model')
    return word_vectors


def get_trained_fasttext_vectors(load_data, cl1, cl2):
    if load_data:
        return pickle.load(open("fasttext.p", "rb"))
    else:
        return train_fasttext_vectors(cl1, cl2, "fasttext")


if __name__ == '__main__':
    cleaned_list_question1 = pickle.load(open("deep_cleaned_list_question1.p", "rb"))
    cleaned_list_question2 = pickle.load(open("deep_cleaned_list_question2.p", "rb"))

    fasttext_word_vectors = get_trained_fasttext_vectors(load_data=True, cl1=cleaned_list_question1, cl2=cleaned_list_question2)
