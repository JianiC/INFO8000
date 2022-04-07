## Perdicti the source country of COVID using WGS
Input: WGS larger than 2000 bp
Output: the potential isolate country of this strain with a pridict probability score

### Data Acquisition
COV-19 sequence were retrieved  from NCBI Genbank with using Bio.Entrez
Using length > 20000bp to filter the whole genome sequences
Using unique to reduced the dataset

### Modeling
Sequence alignment is used with Mafft in python
NLP: to tranl
sklearn.feature_extraction.text import CountVectorizer were used to translate sequence Data
sklearn.naive_bayes import MultinomialNB were used to perdict the spatial information of different sequences
### Data Storage
the training data were stored in a dataframe

### Data Visualization
bar plot showing the spatial distribution of training data : CoV19 sequences

### Web APIs
This is test API to run this perdiction model

local build env FLASK_APP=Api2_test.py flask run

/deploy
deploy the data distribution and accurate score of current model
/
is add:
add new seq to current dataframe
is test:
predict  the isolate country of the sequence
