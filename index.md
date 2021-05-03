# Custom Named Entity Recognition (NER) using SpaCy
### A brief tutorial for beginners like me!


In this tutorial, I want to share what I have learned and apply what I have learned into a specific task, custom Named Entity Recognition (NER) using SpaCy.

Initially, I planned to use the data from [The NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/). However, I decided to use a much [smaller and simpler corpus](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03834-6) to demonstrate the process while finishing the data preprocessing for the NCBI Disease Corpus.

### This tutorial will cover:
1. Installation of spaCy.
2. Preprocessing of the training data.
3. Model training for the custom NER task
4. Model testing using test data

Now, let's get started!

### Installation of spaCy
You can use PIP install to set up spaCyin the Python console by typing:
```markdown
pip install -U pip setuptools wheel
pip install -U spacy
```

### Training data preprocessing 
The training and test data are tsv files. For spaCy to work, the data need to be converted to [binary format](https://spacy.io/api/data-formats#training). To do that, we will first write a function to mannually convert the tsv files to JSON fomart and use ```spacy convert``` to convert the files to the required format.

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













You can use the [editor on GitHub](https://github.com/uazhlt-ms-program/technical-tutorial-wmsun/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/uazhlt-ms-program/technical-tutorial-wmsun/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
