import pandas as pd
import numpy as np

#read csv file
amazon_reviews_read = pd.read_csv('data.csv')

#change column name
amazon_reviews = amazon_reviews_read.rename(columns = {"reviews.text":"reviews",
"reviews.username":"username"})

#drop unwanted columns
to_drop = ['dateAdded','dateUpdated','name','asins','brand','categories','primaryCategories',
'imageURLs','keys','manufacturer','manufacturerNumber','reviews.date','reviews.dateAdded',
'reviews.dateSeen','reviews.doRecommend','reviews.id','reviews.numHelpful','reviews.rating',
'reviews.sourceURLs','reviews.title','sourceURLs' ]
amazon_reviews.drop(to_drop, inplace=True, axis=1)

#check changes
#print(amazon_reviews.head(5))

import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

#remove punctuation
def remove_punctuations(reviews):
    for punctuation in string.punctuation:
        reviews = reviews.replace(punctuation, '')
    return reviews
amazon_reviews['reviews'] = amazon_reviews['reviews'].apply(remove_punctuations)


#tokenize
from nltk.tokenize import word_tokenize
amazon_reviews['reviews'] = amazon_reviews['reviews'].astype(str)
amazon_reviews['reviews'] = amazon_reviews['reviews'].apply(word_tokenize)

#stemming
from nltk.stem import LancasterStemmer
my_stemmer = LancasterStemmer()
amazon_reviews['reviews'] = [[my_stemmer.stem(word) for word in sentence] for sentence in amazon_reviews.reviews]

#remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
amazon_reviews['reviews'] = amazon_reviews['reviews'].apply(lambda x: [item for item in x if item not in stop]) 

#calculate sentiment_score
from textblob import TextBlob
def senti_pol(text):
    for word in text:
        return TextBlob(word).sentiment.polarity  

amazon_reviews['senti_polarity'] = amazon_reviews['reviews'].apply(senti_pol)
 
#dividing sentiments from sentiment polarity score
condition = [
        (amazon_reviews['senti_polarity'] > 0.05),
        (amazon_reviews['senti_polarity'] <= 0.05) & (amazon_reviews['senti_polarity'] > -0.05),
        (amazon_reviews['senti_polarity'] <= -0.05)
            ]
values = ['positive', 'neutral', 'negative']
amazon_reviews['sentiment'] = np.select(condition, values, default='Neutral')

print(amazon_reviews.head(5))

#check if classes are balanced
target_count = amazon_reviews.sentiment.value_counts()

#vec
amazon_reviews['reviews'] = [" ".join(review) for review in amazon_reviews['reviews'].values]
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8)
processed_features = vectorizer.fit_transform(amazon_reviews.reviews).toarray()
y = amazon_reviews['sentiment']

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plot
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
ros = ros.fit(processed_features, y)
X_resampled, y_resampled = ros.fit_resample(processed_features, y)
target_count2 = y_resampled.value_counts()

#splitting train-test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=100)
#NN model implementation
from sklearn.neural_network import MLPClassifier
clf_NN = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
clf_NN.fit(X_train, y_train)
y_pred = clf_NN.predict(X_test)

#Chech NN model performance
from sklearn import metrics
print("Accuracy for Neural Network:", metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred, labels=["negative", "positive", "neutral"])
print(cm)
#svm model performance
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
print(classification_report(y_test,y_pred))