#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import matplotlib;
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from io import BytesIO
import base64
matplotlib.use('Agg')

sns.set()


# In[4]:


app = Flask("cov19",template_folder='templates')
classifier = None
cov19_dna = None
df=None


# In[5]:


#userful function that gets the current figure as a base 64 image for embedding into websites
def getCurrFigAsBase64HTML():
    im_buf_arr = BytesIO()
    plt.gcf().savefig(im_buf_arr,format='png')
    im_buf_arr.seek(0)
    b64data = base64.b64encode(im_buf_arr.read()).decode('utf8');
    return render_template('img.html',img_data=b64data) 


# In[6]:


#convert a sequence of characters into k-mer words, default size = 6 (hexamers)
def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]


# In[7]:


def train():
    global df,cov19_dna,cov19_texts,cv,classifier
    df = pd.read_csv('CoV19seq_country.csv')
    cov19_dna=df
    #convert our training data sequences into short overlapping k-mers of length 6. 
    cov19_dna['words'] = cov19_dna.apply(lambda x: Kmers_funct(x['seq']), axis=1)
    cov19_dna=df.drop("seq",axis=1)
    cov19_texts = list(cov19_dna['words'])
    
    y = cov19_dna.iloc[:, 0].values


    for item in range(len(cov19_texts)):
        cov19_texts[item] = ' '.join(cov19_texts[item])

    
    # convert k-mer words into numerical vectors that represent counts for every k-mer in the vocabulary
    cv = CountVectorizer(ngram_range=(4,4)) #The n-gram size of 4 is previously determined by testing
    X = cv.fit_transform(cov19_texts)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3) 
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    

    pickle.dump(classifier,open("model","wb"))
    pickle.dump(cov19_dna,open("data","wb"))
    return accuracy_score(y_test, y_pred)


# In[14]:


def init():
    global df
    global cov19_dna
    
    df = pd.read_csv('CoV19seq_country.csv')
    cov19_dna=df
   

    
    train()
    


# In[15]:


try:
    classifier = pickle.load(open("model","rb"))
    cov19_dna = pickle.load(open("data","rb"))
except:
    init()


# In[10]:


#this method resets/initializes everything (database, model) (should probably password protect this)
@app.route("/reset")
def reset():
    init()
    return "reset model"
    


# In[11]:


# show an interface to add/test data, which will hit test
@app.route("/")
def main():
    return render_template("main.html")


# In[12]:


@app.route("/run_observation",methods=["POST"])
def add_data():
    global df
    global cov19_dna
    global classifier
    global d
    
    try:
        seq = request.values.get("seq")
        country = request.values.get("country")
        #sepal_width = float(request.values.get('sepal_width',0.0))
        #petal_width = float(request.values.get('petal_width',0.0))

        is_add = request.values.get("add","no")
        is_test = request.values.get("test","no")
    except: 
        return "Error parsing entries"
    
       
    if is_add != "no":
    
        obs = pd.DataFrame([[seq,country]],columns=["seq","country"])
        df2 = pd.read_csv('CoV19seq_country.csv')
        obs2 = pd.concat([obs,df2],ignore_index=True)

        
        
        d=obs2.groupby(['country']).count()
        d.reset_index(level=['country'], inplace=True)
        sns.catplot(data=d,x="country",y="seq",kind="bar")
        img_html = getCurrFigAsBase64HTML();

        
        return "Added new sample " + "<pre>"+ obs2.to_string() + "</pre><br> ... <br> and retrained." + "<br>" + img_html
    
    
    
    if is_test != "no":


        obs = pd.DataFrame([[seq,country]],columns=["seq","country"])
        #obs = pd.DataFrame([[sepal_width,petal_width,]],
                           #columns=["sepal.width","petal.width"])    	

        
        df2 = pd.read_csv('CoV19seq_country.csv')                   
        obs2 = pd.concat([obs,df2],ignore_index=True)
        obs2=obs2.filter(['seq','country'], axis=1)
        #obs2=pd.read_csv('CoV19seq_country.csv')
        
        obs2['words'] = obs2.apply(lambda x: Kmers_funct(x['seq']), axis=1)
        obs2=obs2.drop("seq",axis=1)
        
        obs2_texts = list(obs2['words'])
        
        #separate labels
        y_obs2 = obs2.iloc[:, 0].values	

        for item in range(len(obs2_texts)):
            obs2_texts[item] = ' '.join(obs2_texts[item])

    	# convert k-mer words into numerical vectors that represent counts for every k-mer in the vocabulary

        cv = CountVectorizer(ngram_range=(4,4)) #The n-gram size of 4 is previously determined by testing
        X = cv.fit_transform(obs2_texts)	
        
        def remove_first_a(t):
            return t[1:]
            

        x_train = remove_first_a(X)
        x_test = X[0]
        y_train = remove_first_a(y_obs2)
        
        classifier = MultinomialNB(alpha=0.1)
        classifier.fit(x_train, y_train)
        
        y_pred = classifier.predict(x_test)
        
        

        #return obs.to_string()
        return classifier.predict(x_test)[0]
                
    return "not implemented"
    
   
 


# In[13]:


# this function display the classification model
@app.route("/deploy",methods=['GET','POST'])
def deploy():
    global cov19_dna
    global classifier
    
    
    s = train()
    d=df.groupby(['country']).count()
    d.reset_index(level=['country'], inplace=True)
    sns.catplot(data=d,x="country",y="seq",kind="bar")
    img_html = getCurrFigAsBase64HTML();


    return "accuracy score for the model is " + str(s) + img_html
    


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




