#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Importation des librairies
from flask import Flask, request, render_template, redirect, url_for
from data_science.predict import get_sentiment


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Racine de l'application web
    """

    if request.method == 'POST':
        sentence = request.form.get("sentence", '')
        return redirect(url_for('result', sentence=sentence))
    elif request.method == 'GET':
        return render_template('index.html')
    else:
        raise ValueError("This method is not implemented !")


@app.route('/result?<sentence>')
def result(sentence):
    """
    Page d'affichage des résultats de la prédiction
    """
    sentiment = get_sentiment(sentence)
    return render_template('result.html', sentence=sentence, sentiment=sentiment)


if __name__ == '__main__':

    app.run(port=8000, debug=True)
