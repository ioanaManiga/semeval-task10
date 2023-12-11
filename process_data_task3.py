import json
from math import isnan

import pandas as pd

with open('MELD_train_efr.json') as f:
    task3_train_data = json.load(f)

df = pd.DataFrame(columns=('speakers', 'utterances', 'emotions', 'triggers'))

index = 0
for dialog in task3_train_data:
    for (speaker, utterance, emotion, trigger_value) in zip(dialog["speakers"], dialog["utterances"],
                                                            dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [speaker, utterance, emotion, trigger_value]
        index += 1

X = pd.DataFrame(df, columns=["speakers","utterances", "emotions"])
Y = pd.DataFrame(df, columns=["triggers"])
X_train = [{"speaker":item[0], "utterance":item[1], "emotion":item[2]} for item in X.values]
Y_train = []
for item in Y.values:
    if(not isnan(item[0])):
        Y_train.append(item[0])
    else:
        Y_train.append(0.0)

json_object_X = json.dumps(X_train, indent=4)
with open("task3_train_attributes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_X)


json_object_Y = json.dumps(Y_train, indent=4)
with open("task3_train_classes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_Y)

##########################################################################################################################

with open('MELD_val_efr.json') as f:
    task3_val_data = json.load(f)

df = pd.DataFrame(columns=('speakers', 'utterances', 'emotions', 'triggers'))

index = 0
for dialog in task3_val_data:
    for (speaker, utterance, emotion, trigger_value) in zip(dialog["speakers"], dialog["utterances"],
                                                            dialog["emotions"], dialog["triggers"]):
        df.loc[index] = [speaker, utterance, emotion, trigger_value]
        index += 1

X = pd.DataFrame(df, columns=["speakers", "utterances", "emotions"])
Y = pd.DataFrame(df, columns=["triggers"])
X_test = [{"speaker": item[0], "utterance": item[1], "emotion": item[2]} for item in X.values]
Y_test = []
for item in Y.values:
    if(not isnan(item[0])):
        Y_test.append(item[0])
    else:
        Y_test.append(0.0)

json_object_X = json.dumps(X_train, indent=4)
with open("task3_val_attributes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_X)

json_object_Y = json.dumps(Y_train, indent=4)
with open("task3_val_classes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_Y)