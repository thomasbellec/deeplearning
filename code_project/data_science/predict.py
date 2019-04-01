#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importation des librairies utiles
from code_project.data_science.utils_project import load_saved_file, process_nlp, apply_vectorization
from code_project.config_project import *

import pandas as pd


def get_sentiment(sentence):
    """
    Cette fonction prédit le sentiment de la phrase en argument
    """
    # Chargement du modèle et de la vectorisation enregistrés
    model = load_saved_file(MODEL_PATH)
    vectorizer = load_saved_file(VECTOR_PATH)

    df_sent = pd.DataFrame({'sentence': [sentence]})
    df_sent['nlp_sentence'] = process_nlp(df_sent['sentence'], process_type='stem')
    features = apply_vectorization(df_sent['nlp_sentence'], vectorizer, type_vector=TYPE_VECTOR)

    sentiment = model.predict(features)

    return sentiment[0]
