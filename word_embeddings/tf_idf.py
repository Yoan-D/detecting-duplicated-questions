# Train TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import dill
import pickle


def td_idf_word2weight(list1, list2):
    print("Creating TfidfVectorizer...")
    tfidf = TfidfVectorizer(preprocessor=' '.join)
    tfidf.fit(list1 + list2)

    # if a word was never seen - it is considered to be at least as infrequent as any of the known words
    max_idf = max(tfidf.idf_)
    return collections.defaultdict(
        lambda: max_idf,
        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])


def get_td_idf_word2weight_scores(load_data, cl1, cl2):
    if load_data:
        return dill.load(open("word2weight.p", "rb"))
    else:
        w2w = td_idf_word2weight(cl1, cl2)
        dill.dump(w2w, open("word2weight.p", "wb"))
        print('Done training TfidfVectorizer')
        return w2w


if __name__ == '__main__':
    cleaned_list_question1 = pickle.load(open("deep_cleaned_list_question1.p", "rb"))
    cleaned_list_question2 = pickle.load(open("deep_cleaned_list_question2.p", "rb"))

    word2weight = get_td_idf_word2weight_scores(load_data=True, cl1=cleaned_list_question1, cl2=cleaned_list_question2)
