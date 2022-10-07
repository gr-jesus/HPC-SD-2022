#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)


class ClasificadorNB4Text:
    
    def __init__(
            self, csv_data_file, text_column, clase_column, test_size=0.10):
        
        self._text_column = text_column
        self._clase_column = clase_column
        self._df = pd.read_csv(
            csv_data_file)[[self._text_column, self._clase_column]].dropna()
        self.set_test_size(test_size)
        
    def head(self):
        
        return self._df.head()
    
    def set_test_size(self, test_size=0.10):
        
        self._test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self._df[self._text_column], self._df[self._clase_column],
            random_state=1, test_size=self._test_size)
        
    def create_bag_of_words(self):
        
        self.count_vector = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b")
        self.training_data = self.count_vector.fit_transform(self.X_train)
        self.testing_data = self.count_vector.transform(self.X_test)
        
        palabras = self.count_vector.get_feature_names_out()
        
        try:
            BoW = pd.DataFrame(data=self.training_data.toarray(), columns=palabras)
        except MemoryError:
            BoW = None
        return BoW
    
    def create_and_fit_model(self):
        
        self.naive_bayes = MultinomialNB()
        self.naive_bayes.fit(self.training_data, self.y_train)
    
    def predict(self, data=None):
        
        if not data is None:
            data = self.count_vector.transform(data)
        else:
            data = self.testing_data
        return self.naive_bayes.predict(data)
    
    def evaluate(self, data=None, y_data=None):
        
        predictions = self.predict(data)
        
        if y_data is None:
            y_data = self.y_test
        
        if len(self.naive_bayes.classes_) == 2:
            return {
                'accuracy': accuracy_score(y_data, predictions),
                'precision': precision_score(y_data, predictions),
                'recall': recall_score(y_data, predictions),
                'f1': f1_score(y_data, predictions),
                }
        
        return {
            'accuracy': accuracy_score(
                y_data, predictions),
            'precision': precision_score(
                y_data, predictions, average="weighted"),
            'recall': recall_score(
                y_data, predictions, average="weighted"),
            'f1': f1_score(
                y_data, predictions, average="weighted"),
            }