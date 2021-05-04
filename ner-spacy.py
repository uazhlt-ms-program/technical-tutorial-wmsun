# example
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
import json
import sys
import logging
from spacy.scorer import Scorer
from spacy.gold import GoldParse

# training data preprocessing
def tsv_to_json_format(input_path, output_path, unknown_label):
    try:
        f = open(input_path, 'r')  # input file
        fp = open(output_path, 'w')  # output file
        data_dict = {}
        annotations = []
        label_dict = {}
        s = ''
        start = 0
        for line in f:
            if len(line.strip()) > 0:
                # print("current line:", line)
                if line[0:len(line) - 1] != '.\tO':
                    word, entity = line.split('\t')
                    s += word + " "
                    entity = entity[:len(entity) - 1]
                    if entity != unknown_label:
                        if len(entity) != 1:
                            d = {}
                            d['text'] = word
                            d['start'] = start
                            d['end'] = start + len(word) - 1
                            try:
                                label_dict[entity].append(d)
                            except:
                                label_dict[entity] = []
                                label_dict[entity].append(d)
                    start += len(word) + 1
                else:
                    data_dict['content'] = s
                    s = ''
                    label_list = []
                    for ents in list(label_dict.keys()):
                        for i in range(len(label_dict[ents])):
                            if (label_dict[ents][i]['text'] != ''):
                                l = [ents, label_dict[ents][i]]
                                for j in range(i + 1, len(label_dict[ents])):
                                    if (label_dict[ents][i]['text'] == label_dict[ents][j]['text']):
                                        di = {}
                                        di['start'] = label_dict[ents][j]['start']
                                        di['end'] = label_dict[ents][j]['end']
                                        di['text'] = label_dict[ents][i]['text']
                                        l.append(di)
                                        label_dict[ents][j]['text'] = ''
                                label_list.append(l)

                    for entities in label_list:
                        label = {}
                        label['label'] = [entities[0]]
                        label['points'] = entities[1:]
                        annotations.append(label)
                    data_dict['annotation'] = annotations
                    annotations = []
                    json.dump(data_dict, fp)
                    fp.write('\n')
                    data_dict = {}
                    start = 0
                    label_dict = {}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

tsv_to_json_format("data/train.tsv", 'data/train.json', 'abc')


# TRAIN_DATA = [
#     ('Who is Nishanth?', {
#         'entities': [(7, 15, 'PERSON')]
#     }),
#      ('Who is Kamal Khumar?', {
#         'entities': [(7, 19, 'PERSON')]
#     }),
#     ('I like London and Berlin.', {
#         'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
#     })
# ]
#

# load training data
training_data = []
lines = []
input_file = 'data/train.json'
# output_file = 'data/train'
with open(input_file, 'r') as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    text = data['content']
    entities = []
    for annotation in data['annotation']:
        point = annotation['points'][0]
        labels = annotation['label']
        if not isinstance(labels, list):
            labels = [labels]

        for label in labels:
            entities.append((point['start'], point['end'] + 1, label))

    training_data.append((text, {"entities": entities}))

# print(training_data)


model = None
output_dir=Path("C:/Users/Wenmo/Google-Drive/INFO539/technical-tutorial-wmsun/output/")
n_iter = 100

# with open(output_file, 'wb') as fp:
#     pickle.dump(training_data, fp)


# pip install spacy-lookups-data

TRAIN_DATA = training_data

#load the model

if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

#set up the pipeline

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        # print("shuffled training data: ", TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            # print("text: ", text)
            # print("annotations: ", annotations)
            # print("checker2.2")
            if annotations != {'entities': []}:
                nlp.update(
                    [text],
                    [annotations],
                    drop=0.5,
                    sgd=optimizer,
                    losses=losses)
        print(losses)

for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities in the each of the training data: ', [(ent.text, ent.label_) for ent in doc.ents])

if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

print("end of training.")

# test the trained model
# prepare the test data
tsv_to_json_format("data/test.tsv", 'data/test.json', 'abc')

test_data = []
lines = []
input_test = 'data/test.json'
with open(input_test, 'r') as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    text = data['content']
    entities = []
    for annotation in data['annotation']:
        point = annotation['points'][0]
        labels = annotation['label']
        if not isinstance(labels, list):
            labels = [labels]

        for label in labels:
            entities.append((point['start'], point['end'] + 1, label))

    test_data.append((text, {"entities": entities}))
#
# print("test data")
# print(test_data)
# print("Loading from", output_dir)
# nlp2 = spacy.load(output_dir)
# doc2 = nlp2(test_data)
# for ent in doc2.ents:
#     print(ent.label_, ent.text)

def evaluate(ner_model, test_data):
    scorer = Scorer()
    for input_, annot in test_data:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities = annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

ner_model = spacy.load(output_dir)
results = evaluate(ner_model, test_data)

print(results['ents_per_type'])