# Detecting-duplicated-questions
I tackle the task of detecting semantically similar sentences and classifying them as duplicates. <br />
I used the [Quora dataset](https://www.kaggle.com/c/quora-question-pairs) made available to the public. <br />

The questions in the dataset go through several preprocessing steps before being converted into word embeddings. <br />
I experiment with three embedding approaches: Word2Vec[1,2], Fasttext[3, 4], and Doc2Vec [5]. Additionally, I use a fourth approach by combining Word2Vec and Term Frequency – Inverse Document Frequency (TF-IDF) [6, 7].












### References
[1] Mikolov, T., et al. 2013a. Distributed representations of words and phrases and their compositionality. In Advances in    
Neural Information Processing Systems. <br />
[2] Mikolov, T., et al. 2013. Efficient estimation of word representations in vector space. ICLR Workshop. <br />
[3] Bojanowski, P., et al. 2017. Enriching word vectors with subword information. TACL 5:135–146. <br />
[4] Joulin, A., et al. 2017. Bag of tricks for efficient text classification. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics (EACL). <br />
[5] Le, V. Q, Mikolov, T. 2014. Distributed Represenations of Sentences and Documents. In Proceedings of ICML.
[6] Ramos, J. 2003. Using tf-idf to determine word relevance in document queries. In ICML.
[7] Jurafsky, D. and Martin, H. J. 2018. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. Pearson. Prentice Hall, Third Edition draft.
