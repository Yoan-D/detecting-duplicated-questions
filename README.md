# Detecting-duplicated-questions
I tackle the task of detecting semantically similar sentences and classifying them as duplicates. <br />
I used the [Quora dataset](https://www.kaggle.com/c/quora-question-pairs) made available to the public. <br />

The questions in the dataset go through several preprocessing steps before being converted into word embeddings. <br />
I experiment with three embedding approaches: Word2Vec[1,2], Fasttext[3, 4], and Doc2Vec [5]. Additionally, I use a fourth approach by combining Word2Vec and Term Frequency – Inverse Document Frequency (TF-IDF) [6, 7]. The network used to classify questions pairs as duplicates is based on the CNN architecture for sentence classification introduced by Yoon Kim [8] in 2014. I slightly adapt his original network by adding a cosine similarity layer to the end of the network. A similar approach can be found in [9].

## Text preprocessing

1. Every sentence is stripped of all punctuation.
2. Some typos are fixed manually.
3. Every sentence is tokenized.
4. A pretrained [Word2Vec model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and [Symspellpy](https://github.com/mammothb/symspellpy) are used to fix more typos and to segment words appropriately if necessary.
5. Stop words are not removed since they are semantically important in some cases.

#### Usage
1. Please refer to ```README``` in ```pretrained_word_vectors```to download the pretrained Word2Vec model.
2. Run ```python text_preprocessing.py``` to clean and tokenize the questions

## Text analysis

I examined the word frequency counts in the text corpus derived from the questions.

<p align="center">

<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/most_common_words.png" width="600">
</p>

<br />
<p align="center">

<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/log_frequency_diagram.png" width="600">

</p>

#### Conclusions from the text analysis

- Since words that occur only once or twice make up 43.21% of the text corpus, they cannot be discarded when training the embedding models.

- The Skip-Gram model will be used to create word embeddings because, as Mikolov et al. point out [1], it works well on small data sets and is more susceptible to rare words.

- Interestingly, most of the the word occurrences can be found between 3 standard deviations above the mean and 1 standard deviation below the mean.

- Words such as "the" and "what", which occur very often, can be found between 5 and 6 standard deviations above the mean. However, as the histogram shows, they are exceptions.

#### Usage

1. Make sure to run the preprocessing steps first.
2. Run ```python text_analyzing.py``` 

## Creating word embeddings
As previously mentioned I used four different word embedding approaches. <br />
Word embeddings can be created for each model by running the scripts in ```word_embeddings``` <br />
Word2Vec, Doc2Vec and Fasttext are all trained with the same parameters.

#### Usage

1. Make sure to run the preprocessing steps first.
2. For Word2Vec run ```python word2vec.py``` 
3. For Doc2Vec run ```python doc2vec.py``` 
4. For Fasttext run ```python fasttext.py``` 
4. For TF-IDF run ```python tf_idf.py``` 

## Visualizing word embeddings
I visualize the word embeddings created by Word2Vec, Doc2Vec, and Fasttext by reducing their dimensionalities with T-distributed Stochastic Neighbor Embedding (t-SNE) [10]. Credits to [jeffd23](https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne). In the examples depicted below, I use the trained models to find the top 10 similar words for the words *car*, *guns*, *isis*, and *trump*. It is really interesting to see how words are mapped in vector space and which ones are similar according to the cosine similarity measurement. <br />


<p align="center">

<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/doc2vec_car.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/word2vec_car.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/fasttext_car.png" width="270">

</p>

<p align="center">

<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/doc2vec_guns.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/word2vec_guns.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/fasttext_guns.png" width="270">

</p>

<p align="center">

<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/doc2vec_isis.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/word2vec_isis.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/fasttext_isis.png" width="270">

</p>

<p align="center">

<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/doc2vec_trump.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/word2vec_trump.png" width="270">
<img src="https://github.com/Yoan-D/detecting-duplicated-questions/blob/master/screenshots/fasttext_trump.png" width="270">

</p>
#### Usage
1. The visualizing script can be run in the provided jupyter notebook.


## References
[1] Mikolov, T., et al. 2013a. Distributed representations of words and phrases and their compositionality. In Advances in    
Neural Information Processing Systems. <br />
[2] Mikolov, T., et al. 2013. Efficient estimation of word representations in vector space. ICLR Workshop. <br />
[3] Bojanowski, P., et al. 2017. Enriching word vectors with subword information. TACL 5:135–146. <br />
[4] Joulin, A., et al. 2017. Bag of tricks for efficient text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL). <br />
[5] Le, V. Q, Mikolov, T. 2014. Distributed Represenations of Sentences and Documents. In Proceedings of ICML. <br />
[6] Ramos, J. 2003. Using tf-idf to determine word relevance in document queries. In ICML. <br />
[7] Jurafsky, D. and Martin, H. J. 2018. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. Pearson. Prentice Hall, Third Edition draft. <br />
[8] Kim, Y. 2014. Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.<br />
[9] Bogdanova D., dos Santos C. N., Barbosa L., and Zadrozny B. 2015. Detecting semantically equivalent questions in online user forums. In Proceedings of the 19th Conference on Computational Natural Language Learning, CoNLL 2015, Beijing, China, pages 123–131. <br />
[10] Heuer, H. 2015.Text comparison using word vector representations and dimensionality reduction. Proc. EuroSciPy, pp. 13-16.

