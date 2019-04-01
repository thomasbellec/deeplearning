#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importation des librairies utiles
from code_project.data_science.preprocessing import PreProcesseur
from code_project.config_project import *
import pickle
import os
import pandas as pd
import numpy as np


def load_saved_file(filename):
    """
    Charge un fichier enregistré au niveau d'un certain path
    """
    if not os.path.exists(filename):
        raise ValueError("No file saved at {}".format(filename))
    content = pickle.load(open(filename, 'rb'))
    return content


def read_feel(path_feel, limit=None):
    """
    Cette fonction lit les données du dataset feel et retourne un dataframe pandas
    """
    if not os.path.exists(path_feel):
        raise FileNotFoundError("file {} doesn't exist".format(path_feel))

    feel = pd.read_csv(path_feel, sep=";", header=None)
    feel.columns = ['id', 'sentence', 'valence', 'rien', 'fear', 'sadness', 'anger', 'surprise', 'disgust', 'joy']
    feel = feel.drop(['id', 'valence', 'rien'], axis=1)

    if limit is not None:
        feel = feel[:limit]

    return feel


def process_label(dataset, only_sentiment=False):
    """
    Processing impliquant la duplication des lignes qui possèdent plusieurs sentiments
    only_sentiment est une option permettant d'enlever le 'no_emotion'
    """
    new_dataset = pd.DataFrame(columns=['sentence', 'emotion'])
    list_emotion = ['fear', 'sadness', 'anger', 'surprise', 'disgust', 'joy']

    for i in range(dataset.shape[0]):
        has_emotion = False
        for sentiment in list_emotion:
            if dataset.iloc[i][sentiment] == 1:
                has_emotion = True
                new_dataset = new_dataset.append({'sentence': dataset.iloc[i]['sentence'],
                                                  'emotion': sentiment
                                                  }, ignore_index=True)
        if not has_emotion and not only_sentiment:
            new_dataset = new_dataset.append({'sentence': dataset.iloc[i]['sentence'],
                                              'emotion': 'no_emotion'
                                              }, ignore_index=True)

    return new_dataset


def process_nlp(data, process_type='stem', feel=True):
    """
    Traitement NLP du dataset (normalisation, tokenisation, stemming)
    """
    preprocessor = PreProcesseur()
    data = data.map(lambda sentence: preprocessor.process_all(sentence, type=process_type, feel=feel))

    return data


def vectorize_tfidf(data):
    """
    Vectorise des phrases traitées selon la méthode TF-IDF
    """
    x_train = [" ".join(sentence) for sentence in data]

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(CountVectorizer(), TfidfTransformer())

    pipe.fit(x_train)
    feat_train = pipe.transform(x_train)

    return feat_train, pipe


def vectorize_w2v(data, max_epochs=100):
    """
    Vectorise des phrases traitées de l'embbeding gensim (Word2Vec)
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from nltk.tokenize import word_tokenize

    data = data.map(lambda el: " ".join(el))

    # Définition du document
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
                   for i, _d in enumerate(data)]

    # Définition du modèle
    model = Doc2Vec(size=300,
                    alpha=0.025,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1,
                    workers=3)

    # Construction du vocabulaire
    model.build_vocab(tagged_data)

    # Entrainement du modèle
    for epoch in range(max_epochs):
        print('iteration {}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    # Calcul des vecteurs
    feat_train = np.vstack([model.infer_vector(doc.words, steps=20) for doc in tagged_data])

    return feat_train, model


def vectorize(data, type_vector):
    """
    Vectorise les phrases du dataset selon une certaine méthode
    """

    if type_vector == 'w2v':
        feat_train, vectorizer = vectorize_w2v(data, max_epochs=100)
    elif type_vector == 'tf-idf':
        feat_train, vectorizer = vectorize_tfidf(data)
    else:
        raise ValueError("Method not implemented")

    # Enregistrement du modèle
    pickle.dump(vectorizer, open(VECTOR_PATH, 'wb'))

    return feat_train, vectorizer


def apply_vectorization(data, vectorizer, type_vector):
    """
    Applique la vectorisation entrainée à la nouvelle phrase
    """

    if type_vector == 'w2v':
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from nltk.tokenize import word_tokenize
        data = data.map(lambda el: " ".join(el))
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()),
                                      tags=[str(i)]) for i, _d in enumerate(data)]
        features = np.vstack([vectorizer.infer_vector(doc.words, steps=20) for doc in tagged_data])

    elif type_vector == 'tf-idf':
        data = data.map(lambda el: " ".join(el))
        features = vectorizer.transform(data)

    else:
        raise ValueError("Method not implemented")

    return features
