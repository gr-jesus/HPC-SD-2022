#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ClasificadorNB4Text import ClasificadorNB4Text

clasificador = ClasificadorNB4Text(
    'tiktok/tiktok_google_play_reviews.csv', 'content', 'score')

print(clasificador.head(), '\n\n')

clasificador.create_bag_of_words()
clasificador.create_and_fit_model()

ev = clasificador.evaluate()

print(f' Accuracy score: {ev["accuracy"]}')
print(f'Precision score: {ev["precision"]}')
print(f'   Recall score: {ev["recall"]}')
print(f'       F1 score: {ev["f1"]}')
