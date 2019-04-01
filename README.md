# Projet Analyse de Sentiment

_Groupe 28 :_  
- _Thomas Bellec_
- _Alix Mallard_

### Objectif

Ce projet a pour but de présenter une étude d'analyse de sentiments en français.
Nous nous servons ici de deux datasets : FEEL et EMOTION IN TEXT.

### Structure

Ce projet se décompose de la manière suivante : 
- code_projet
    - data_science 
        - autre_dataset.ipynb : jupyter notebook permettant de tester des modèles sur le dataset EMOTION IN TEXT.
        - Dataset_Feel.ipynb : jupyter notebook permettant de tester des modèles sur le dataset FEEL.
        - evaluate.py : script permettant d'évaluer le modèle choisi dans config.py sur le dataset de test
        - learning.py : script permettant l'apprentissage d'un modèle
        - predict.py : script permettant la prédiction d'une phrase à partir d'un modèle appris
        - preprocessing.py : ensemble de traitement NLP permettant de traiter nos phrases
        - utils_project.py : ensemble de fonctions utiles pour le traitement de nos données
    - config.py : fichier de configuration contenant les variables globales du projet
- data
    - Feel-fr : ensemble de fichiers contenant en particulier le dataset FEEL
    - kaggle-traduction : ensemble de fichiers contenant en particulier le dataset EMOTION IN TEXT traduit
    - modeles : dossier regroupant les fichiers de sauvegarde des apprentissages et vectorisations
    - test_set.xlsx : dataset de test
- requirements.txt : fichier contenant les modules nécessaires au projet
