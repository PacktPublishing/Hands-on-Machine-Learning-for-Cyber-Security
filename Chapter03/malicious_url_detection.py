import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#The url needs to undergo some amount of cleasing before we use it. We tokenize it by removing slash , dots and coms
def url_cleanse(web_url):                      
    web_url = web_url.lower()
    urltoken = []
    dot_slash = []
    slash = str(web_url).split('/')
    for i in slash:
        r1 = str(i).split('-')            
        token_slash = []
        for j in range(0,len(r1)):
            r2 = str(r1[j]).split('.')  
            token_slash = token_slash + r2
        dot_slash = dot_slash + r1 + token_slash 
    urltoken = list(set(dot_slash))       
    if 'com' in urltoken:
        urltoken.remove('com')               
    return urltoken

# We injest the data and convert it to the relevant dataframes.
input_url = '~/data.csv'
data_csv = pd.read_csv(input_url,',',error_bad_lines=False)
data_df = pd.DataFrame(data_csv)                                                                                                
url_df = np.array(data_df)                      
random.shuffle(data_df)
y = [d[1] for d in data_df]                  
inputurls = [d[0] for d in data_df]               
#http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/

#We need to generate the tf-idf from the urls.

url_vectorizer = TfidfVectorizer(tokenizer=url_cleanse)  
x = url_vectorizer.fit_transform(inputurls)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

l_regress = LogisticRegression()                  # Logistic regression
l_regress.fit(x_train, y_train)
l_score = l_regress.score(x_test, y_test)
print("score: {0:.2f} %".format(100 * l_score))
url_vectorizer_save = url_vectorizer


file1 = "model.pkl"
with open(file1, 'wb') as f:
    pickle.dump(l_regress, f)
f.close()

file2 = "vector.pkl"
with open(file2,'wb') as f2:
    pickle.dump(vectorizer_save, f2)
f2.close()

#We load a bunch of urls that we want to check are legit or not
urls = ['hackthebox.eu','facebook.com']

file1 = "model.pkl"
with open(file1, 'rb') as f1:  
    lgr = pickle.load(f1)
f1.close()
file2 = "pvector.pkl"
with open(file2, 'rb') as f2:  
    url_vectorizer = pickle.load(f2)
f2.close()
url_vectorizer = url_vectorizer
x = url_vectorizer.transform(inputurls)
y_predict = l_regress.predict(x)
print(inputurls)
print(y_predict)

# We can use the whitelist to make the predictions
whitelisted_url = ['hackthebox.eu','root-me.org']
some_url = [i for i in inputurls if i not in whitelisted_url]

file1 = "model.pkl"
with open(file1, 'rb') as f1:  
    l_regress = pickle.load(f1)
f1.close()
file2 = "vector.pkl"
with open(file2, 'rb') as f2:  
    url_vectorizer = pickle.load(f2)
f2.close()
url_vectorizer = url_vectorizer
x = url_vectorizer.transform(some_url)
y_predict = l_regress.predict(x)

for site in whitelisted_url:
    some_url.append(site)
print(some_url)
l_predict = list(y_predict)
for j in range(0,len(whitelisted_url)):
    l_predict.append('good')
print(l_predict)


#use SVM
from sklearn.svm import SVC
svmModel = SVC()
svmModel.fit(X_train, y_train)
#lsvcModel = svm.LinearSVC.fit(X_train, y_train)
svmModel.score(X_test, y_test)

file1 = "model.pkl"
with open(file1, 'rb') as f1:  
    svm_model = pickle.load(f1)
f1.close()
file2 = "pvector.pkl"
with open(file2, 'rb') as f2:  
    url_vectorizer = pickle.load(f2)
f2.close()

test_url = "http://www.isitmalware.com"
vec_test_url = url_vectorizer.transform([trim(test_url)])
result = svm_model.predict(vec_test_url)
print(test_url)
print(result)



