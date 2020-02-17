# Train doc2vec model
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import logging
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')


def doc2vec(doc_cleaned_list_question1, doc_cleaned_list_question2, train):
    labeled_questions = []
    for index, row in train.iterrows():
        if type(row['qid1']) is not int or type(row['qid2']) is not int:
            print('fail')
        labeled_questions.append(TaggedDocument(doc_cleaned_list_question1[index], [row['qid1']]))
        labeled_questions.append(TaggedDocument(doc_cleaned_list_question2[index], [row['qid2']]))
    doc2vec_model = Doc2Vec(dm=1, min_count=1, window=5, vector_size=300, sample=1e-4, negative=10)
    doc2vec_model.build_vocab(labeled_questions)
    doc2vec_model.wv.init_sims(replace=True)
    try:
        doc2vec_model.train(labeled_questions, epochs=20, total_examples=doc2vec_model.corpus_count)
    except TypeError:
        print("Error occurred")
    pickle.dump(doc2vec_model, open("doc2vec_model" + ".p", "wb"))
    print('Done training doc2vec')
    return doc2vec_model


def get_trained_document_vectors(load_data, cl1, cl2, data):
    if load_data:
        return pickle.load(open("doc2vec_model.p", "rb"))
    else:
        return doc2vec(cl1, cl2, data)


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')

    cleaned_list_question1 = pickle.load(open("deep_cleaned_list_question1.p", "rb"))
    cleaned_list_question2 = pickle.load(open("deep_cleaned_list_question2.p", "rb"))

    doc_vectors = get_trained_document_vectors(load_data=True, cl1=cleaned_list_question1, cl2=cleaned_list_question2, data=train)
