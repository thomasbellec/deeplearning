#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importation des librairies
import re
import unicodedata
import time
from nltk import word_tokenize
from nltk.corpus import stopwords
from num2words import num2words
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer


class PreProcesseur:

    """
    Classe permettant de pré-processer les données textuelles pour l'analyse de sentiment
    """

    def __init__(self):
        self._lang = 'fr'

    # Méthodes principales de la classe
    def tokenisation(self, sentence):
        """Convert apostrophe (for french) and tokenize sentence"""
        if self._lang == 'fr':
            sentence = sentence.replace("'", " ")  # en français on remplace les apostrophes par des espaces
            sentence = sentence.replace("-", " ")
            return word_tokenize(sentence, language='french')
        else:
            return word_tokenize(sentence)

    def normalize(self, words, feel=True):
        """Normalize words from list of tokenized words"""
        stop_words = set(stopwords.words('french'))
        new_words = []
        rm_pseudo = False
        for word in words:
            if rm_pseudo:
                rm_pseudo = False
                continue
            if not feel:
                rm_pseudo = self._remove_pseudo(word)
            word = self._remove_website(word)
            word = self._remove_non_ascii(word)
            word = self._to_lowercase(word)
            word = self._remove_punctuation(word)
            word = self._replace_numbers(word)
            word = self._remove_stopwords(word, stop_words)
            if word != "":
                new_words.append(word)
        return new_words

    def stem_words(self, words):
        """Stem words in list of tokenized words"""
        if self._lang == 'fr':
            stemmer = FrenchStemmer()
        else:
            stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words"""
        if self._lang == 'fr':
            lemmatizer = FrenchLefffLemmatizer()
        else:
            lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def process_all(self, sentence, type="stem", verbose=False, feel=True):
        """Process whole pipeline from a sentence"""
        t1 = time.time()
        w_tok = self.tokenisation(sentence)
        t2 = time.time()
        if verbose:
            print("{}s for tokenisation".format(t2-t1))
        w_norm = self.normalize(w_tok, feel=feel)
        t3 = time.time()
        if verbose:
            print("{}s for normalization".format(t3-t2))

        if type == "lemmatize":
            tokens = self.lemmatize_verbs(w_norm)
        elif type == "stem":
            tokens = self.stem_words(w_norm)
        elif type is None:
            tokens = w_norm
        else:
            raise ValueError("wrong type, it should be 'lemmatize' or 'stemm'")
        t4 = time.time()
        if verbose:
            print("{}s for stemming".format(t4-t3))

        return tokens

    # Méthodes privées
    @staticmethod
    def _remove_pseudo(word):
        """Remove word if it's a twitter pseudo"""
        if len(word) > 0 and (word[0] == "@" or word[0] == '&'):
            return True
        else:
            return False

    @staticmethod
    def _remove_website(word):
        """Remove word if it's a website"""
        if (len(word) > 2 and (word[:3] == "www" or word[:2] == "//")) or word == 'http':
            return ""
        else:
            return word

    @staticmethod
    def _remove_non_ascii(word):
        """Remove non-ASCII characters from a word"""
        return unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    @staticmethod
    def _to_lowercase(word):
        """Convert all characters to lowercase from a word"""
        return word.lower()

    @staticmethod
    def _remove_punctuation(word):
        """Remove punctuation from a word"""
        return re.sub(r'[^\w\s]', '', word)

    @staticmethod
    def _replace_numbers(word):
        """Replace numbers by letters in word"""
        if word.isdigit():
            return num2words(float(word), lang='fr')
        else:
            return word

    @staticmethod
    def _remove_stopwords(word, stop_words):
        """Remove french stop words from a word"""
        not_stopwords = {'n', 'pas', 'ne'}
        if word not in stop_words or word in not_stopwords:
            return word
        else:
            return ""

    @property
    def lang(self):
        return self._lang


if __name__ == '__main__':

    """ Exemple d'utilisation """

    feel = False
    phrase = "y a t-il quelqu&#39;un?"
    print("Initial : {}".format(phrase))

    preprocess = PreProcesseur()
    w_tok = preprocess.tokenisation(phrase)
    print("\nafter tokenisation : {}".format(w_tok))
    w_norm = preprocess.normalize(w_tok, feel=feel)
    print("after normalization : {}".format(w_norm))

    stems = preprocess.stem_words(w_norm)
    lemmas = preprocess.lemmatize_verbs(w_norm)

    print("\nstems : {}".format(stems))
    print("lemmas : {}\n".format(lemmas))

    tokens = preprocess.process_all(phrase, type='stem', verbose=True, feel=feel)

