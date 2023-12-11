import json

import keras_nlp
import keras
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from IPython.display import display

import pandas as pd
from transformers import DistilBertTokenizer

with open('task3_train_attributes.json') as f:
    task3_train_attributes = json.load(f)

with open('task3_train_classes.json') as f:
    task3_train_classes = json.load(f)

with open('task3_val_attributes.json') as f:
    task3_val_attributes = json.load(f)

with open('task3_val_classes.json') as f:
    task3_val_classes = json.load(f)

v = DictVectorizer(sparse=False)

#############################################################################3

# task3_train_attributes = task3_train_attributes[:100]
# task3_train_classes = task3_train_classes[:100]
# v.fit(task3_train_attributes)
X_train = v.fit_transform(task3_train_attributes)
support = SelectKBest(chi2, k=350).fit(X_train, task3_train_classes)

v.restrict(support.get_support())
X_train = v.transform(task3_train_attributes)
X_train = torch.tensor(X_train)

padding_masking = X_train > 0

padding_masking = padding_masking.int()

train_features = {
    "token_ids": X_train,
    "padding_mask": padding_masking
}

##########################################################################

# task3_val_attributes = task3_val_attributes[:100]
# task3_val_classes = task3_val_classes[:100]
# v.fit(task3_val_attributes)
X_test = v.fit_transform(task3_val_attributes)
support = SelectKBest(chi2, k=350).fit(X_test, task3_val_classes)

v.restrict(support.get_support())
X_test = v.transform(task3_val_attributes)
X_test = torch.tensor(X_test)

padding_masking1 = X_test > 0

padding_masking1 = padding_masking1.int()


test_features = {
    "token_ids": X_test,
    "padding_mask": padding_masking1
}

##########################################################################

# preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
#     "distil_bert_base_en_uncased",
#     sequence_length=128,
# )
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    preprocessor=None,
    num_classes=2
)

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
    metrics=['accuracy', f1_m, precision_m, recall_m]
)

classifier.fit(x=train_features, y=task3_train_classes)

loss, accuracy, f1_score, precision, recall = classifier.evaluate(test_features, task3_val_classes, verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy))
print("Testing F1: {:.4f}".format(f1_score))
print("Testing Precision: {:.4f}".format(precision))
print("Testing Recall: {:.4f}".format(recall))
# loss, accuracy = classifier.evaluate(X_val, Y_val, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))

# print("PREDICTIONS",predictions)
# print("recall",accuracy_score(y_test.astype(int), predictions))
# # print("f1",f1_score(y_test.astype(int), predictions))
# print("precision",precision_score(y_test.astype(int), predictions))
# print("recall",recall_score(y_test.astype(int), predictions))
#
# print(y_test.astype(int))
