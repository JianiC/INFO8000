## INFO 8000 term project
### Data Acquisition
COV-19 sequence were retrieved  from NCBI Genbank with using Bio.Entrez
Using length > 20000bp to filter the whole genome sequences

### Modeling
Sequence alignment is used with Mafft in python

sklearn.feature_extraction.text import CountVectorizer
were used to ranslate sequence Data


sklearn.naive_bayes import MultinomialNB were used
 to perdict the spatial information of different sequences



### Data Storage
data were stored in a dataframe

### Data Visualization
bar plot showing the spatial distribution of CoV19 sequences

### Web APIs
local build env FLASK_APP=Api2_test.py flask run

/deploy
deploy the data distribution and accurate score of current model

/
is add:
add new seq to current dataframe
is test:
predict  the isolate country of the sequence
