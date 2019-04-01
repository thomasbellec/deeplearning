#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

# Définition des paramètres et variables du projet

# Paramètres modifiables
MODEL_NAME = "log_reg_3.sav"             # nom du modèle de machine learning à utiliser
VECTOR_NAME = "tfidf_3.sav"                # nom de la vectorisation à utiliser
TYPE_VECTOR = "tf-idf"                    # type de vecteur à utiliser pour la vectorisation
LIMIT = None                           # nombre de lignes max à utiliser, None pour tout
DATASET_FEEL = True                     # si False on prend le dataset issu de twitter


# Paramètres non modifiables
CWD_PATH = os.path.abspath(os.path.dirname(__file__))
FEEL_PATH = os.path.join(os.path.dirname(CWD_PATH), "data", "Feel-fr", "FEEL.txt")
SAVE_DIR = os.path.join(os.path.dirname(CWD_PATH), "data", "modeles")
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
VECTOR_PATH = os.path.join(SAVE_DIR, VECTOR_NAME)
TEST_SET_PATH = os.path.join(os.path.dirname(CWD_PATH), "data", "test_set.xlsx")


if __name__ == '__main__':

    print('CWD_PATH : ', CWD_PATH)
    print('FEEL_PATH : ', FEEL_PATH)
    print('SAVE_DIR : ', SAVE_DIR)
    print('MODEL_PATH : ', MODEL_PATH)
    print('VECTOR_PATH : ', VECTOR_PATH)


