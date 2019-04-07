# Projet Analyse de Sentiment

_Groupe 28 :_  
- _Thomas Bellec_
- _Alix Mallard_

### Objectif

Ce projet a pour but de présenter une étude d'analyse de sentiments de tweets en français.
Nous nous servons ici de deux datasets : FEEL et EMOTION IN TEXT.

### Structure

Ce projet se décompose de la manière suivante : 
- code_projet
    - data_science 
        - Projet Deep Learning.ipynb : jupyter notebook contenant le travail côté Deep Learning
        - evaluate.py : script permettant d'évaluer le modèle choisi dans config.py sur le dataset de test
        - learning.py : script permettant l'apprentissage d'un modèle
        - predict.py : script permettant la prédiction d'une phrase à partir d'un modèle appris
        - preprocessing.py : ensemble de traitement NLP permettant de traiter nos phrases
        - read_feel.ipynb : jupyter notebook pour l'analyse du dataset FEEL
        - utils_project.py : ensemble de fonctions utiles pour le traitement de nos données
    - config.py : fichier de configuration contenant les variables globales du projet
- data
    - Feel-fr : ensemble de fichiers contenant en particulier le dataset FEEL
    - kaggle-traduction : ensemble de fichiers contenant en particulier le dataset EMOTION IN TEXT traduit
    - test_set.xlsx : dataset de test
- requirements.txt : fichier contenant les modules nécessaires au projet

### Installation
1. Installer le package `virtualenv` (attention il doit être en Python 3.6):

`pip3 install virtualenv` ou `pip install virtualenv`

2. Allez dans le répertoire de votre choix et là, créez un environnement virtuel python dans ledit répertoire et activez-le :

`cd ./your_directory`

`virtualenv venv`

Linux / MacOS: `source venv/bin/activate`

Windows: `.\venv\Scripts\activate.bat`

3. Allez dans le dossier venv et clonez le projet github :

`cd ./venv`

`git clone git@github.com:thomasbellec/deeplearning.git`

4. Pour installer les requirements :

`cd ./deeplearning`

`pip3 install -r requirements.txt or pip install -r requirements.txt`

Ainsi qu'une bibliothèque un peu spécifique : 

`pip install git+https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git`

