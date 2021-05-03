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
The training and test data are tsv files. For spaCy to work, the data need to be converted to binary format that 



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
