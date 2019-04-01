#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importation des librairies
from code_project.data_science.utils_project import read_feel, process_label, process_nlp, vectorize
from code_project.config_project import *

import pickle
import time


def train_model(data_train, y_train):
    """
    Entraine et sauvegarde le modèle sélectionné
    """
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    # model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model = LogisticRegression(C=5, multi_class='auto', penalty='l1', n_jobs=-1)
    model.fit(data_train, y_train)

    pickle.dump(model, open(MODEL_PATH, 'wb'))


def main(verbose=True):

    # Lecture des données
    t1 = time.time()
    feel = read_feel(FEEL_PATH, limit=LIMIT)
    t2 = time.time()
    if verbose:
        print("Dataset read, duration {}s".format(round(t2-t1, 4)))

    # On traite les labels et les données textes
    feel = process_label(feel)
    feel['nlp_sentence'] = process_nlp(feel['sentence'], process_type='stem')
    t3 = time.time()
    if verbose:
        print("Label and NLP process done, duration {}s".format(round(t3-t2, 4)))

    # Séparation des features et de la target
    y_train = feel.emotion
    x_train = feel.nlp_sentence

    # Vectorisation des données
    feat_train, _ = vectorize(x_train, type_vector=TYPE_VECTOR)
    t4 = time.time()
    if verbose:
        print("Vectorization done, duration {}s".format(round(t4-t3, 4)))

    # Entrainement du modèle et sauvegarde de celui-ci
    train_model(feat_train, y_train)
    t5 = time.time()
    if verbose:
        print("Model trained and saved, duration {}s".format(t5-t4, 4))


if __name__ == '__main__':

    main(verbose=True)






