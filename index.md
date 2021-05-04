# Custom Named Entity Recognition (NER) using SpaCy
### A brief tutorial for beginners like me!


In this tutorial, I want to share what I have learned and apply what I have learned into a specific task, custom Named Entity Recognition (NER) using SpaCy.

Initially, I planned to use the data from [The NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/). However, I decided to use a much [smaller and simpler corpus](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03834-6) to demonstrate the process while finishing the data preprocessing for the NCBI Disease Corpus.

### This tutorial will cover:
1. Install spaCy
2. Pre-process the training data
3. Train a model for the custom NER task
4. Test the performance of the model

Now, let's get started!

### Install spaCy
You can use PIP install to set up spaCyin the Python console by typing:
```markdown
pip install -U pip setuptools wheel
pip install -U spacy
```

### Pre-process the training data
The training and test data are tsv files. For spaCy to work, the data need to be converted to [binary format](https://spacy.io/api/data-formats#training). To do that, we will first write a function to mannually convert the tsv files to JSON fomart and use ```spacy convert``` to convert the files to the required format. In this tutorial, instead of using that approach, we can write a couple more lines of code to get the desired format.

The code for converting ```tsv``` to ```JSON``` is shown as below:
```
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

```

Once this step is complete.The ```.json``` file in the data folder will be ready for next step which is to convert the data file into the format spaCy can take.

```
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
```
### Train a model for the custom NER task
Once the data is ready, we can start the training process as below:
```
model = None
output_dir=Path("C:/Users/Wenmo/Google-Drive/INFO539/technical-tutorial-wmsun/output/")
n_iter = 100

TRAIN_DATA = training_data

if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")
    
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
```

### Test the performance of the model

Now, we have built a model for the NER task using spaCy. What we need to do next is to use the test data and see how well the model performs.

```
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
```

Finally, it will print the precision, recall, and F1 score for the model on test data.

At 100 iterations, the trained model's performance on test data:

|   | B-Disease | I-Disease |
| ------------- | ------------- | ------------- |
| Precision  | 0.5  | 0.33 |
| Recall  | 0.25  | 0.14 |
| F1  | 0.33  | 0.2 |

The low scores are not surprising because the training data is a small corpus. However, at least it shows that this method worked. With a larger training corpus, I believe the performance will be greatly improved.


*This tutorial allowed me to explore the NER function of spaCy, and GitHub Pages for the first time. There was a little bit of learning curve but it was a great learning experience.*

*Spring 2021*
