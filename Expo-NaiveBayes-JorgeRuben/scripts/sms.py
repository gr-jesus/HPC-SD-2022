#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ClasificadorNB4Text import ClasificadorNB4Text

clasificador = ClasificadorNB4Text(
    'smsspamcollection/SMSSpamCollection.csv', 'sms_message', 'label')

print(clasificador.head(), '\n\n')
clasificador._df['label'] = clasificador._df.label.map({'ham': 0, 'spam': 1})
clasificador.set_test_size()
print(clasificador.head(), '\n\n')

bow = clasificador.create_bag_of_words()
clasificador.create_and_fit_model()

ev = clasificador.evaluate()

print(f' Accuracy score: {ev["accuracy"]}')
print(f'Precision score: {ev["precision"]}')
print(f'   Recall score: {ev["recall"]}')
print(f'       F1 score: {ev["f1"]}')

print("\nPorcion del BoW")
print(bow[bow.columns[5343:5348]].sort_values(['necessary']))
