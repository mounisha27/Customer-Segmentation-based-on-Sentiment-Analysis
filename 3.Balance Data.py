import pandas as pd
import numpy as np

#read csv file
amazon_reviews_read = pd.read_csv('C:/Users/mouni/Downloads/Thesis Codes/data.csv')

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
print(amazon_reviews.head(5))

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

#check if classes are balanced
print("Before Sampling")
target_count = amazon_reviews.sentiment.value_counts()
print('Class 0:', target_count.iloc[0])
print('Class 1:', target_count.iloc[1])
print('Class 2:', target_count.iloc[2])

target_count.plot(kind='bar', title='Count')

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
print("After Sampling")
print('Class 0:', target_count2.iloc[0])
print('Class 1:', target_count2.iloc[1])
print('Class 2:', target_count2.iloc[2])

print(target_count2.plot(kind='bar', title='Count'))

