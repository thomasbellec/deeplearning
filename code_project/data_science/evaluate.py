#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importation des librairies
from code_project.data_science.utils_project import load_saved_file, process_nlp, apply_vectorization
from code_project.config_project import *

import pandas as pd


def load_test_set(path_test_set):
    """
    Charge le dataset de test
    """
    if not os.path.exists(path_test_set):
        raise FileNotFoundError("file {} doesn't exist".format(path_test_set))
    data_test = pd.read_excel(path_test_set)
    data_test = data_test[['phrase', 'emotion']]

    return data_test


def evaluate_model(display=True):
    """
    Evalue un modèle
    """
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Chargement du modèle et de la vectorisation enregistrés
    model = load_saved_file(MODEL_PATH)
    vectorizer = load_saved_file(VECTOR_PATH)

    # Chargement du dataset de test
    data_test = load_test_set(TEST_SET_PATH)

    # Traitement des données textuelles
    data_test['nlp_sentence'] = process_nlp(data_test['phrase'])

    # Séparation des features et du label
    y_test = data_test.emotion
    x_test = data_test.nlp_sentence

    # Vectorisation des features
    features = apply_vectorization(x_test, vectorizer, type_vector=TYPE_VECTOR)

    # Prédictions à partir du modèle
    predictions = model.predict(features)

    # calcul des performances
    accuracy = accuracy_score(y_test, predictions)
    conf_mat = confusion_matrix(y_test, predictions)

    if display:
        print('EVALUATION FOR MODEL {} with vectorization {}: '.format(MODEL_NAME, VECTOR_NAME))
        print('accuracy : {}'.format(round(accuracy, 3)))
        print('confusion matrix : \n{}'.format(conf_mat))

    return accuracy


if __name__ == '__main__':

    evaluate_model(display=True)