# List of examples of the use of the NLP and Sentiment Analysis in science or practice with references to the literature

- Virtual assistants like Siri, Alexa and Google Assistant
- Translation tools like Gooogle Translate
- Customer service / feedback chatbots
- Product reviews analysis
- Market sentiment (news and social media)

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics. 2011.

Muhammad Taimoor Khan, Mehr Durrani, Armughan Ali, Irum Inayat, Shehzad Khalid & Kamran Habib Khan. Sentiment analysis and the complex natural language. Complex Adaptive Systems Modeling. Volume 4, article number 2. 2016.

Anuja P Jain, Padma Dandannavar. Application of machine learning techniques to sentiment analysis. International Conference on Applied and Theoretical Computing and Communication Technology (iCATccT). 2016.

# Indication of the selected library, functions and functions’ parameters

- Natural Language Toolkit - https://www.nltk.org/
- Porter Stemmer algorithm - https://tartarus.org/martin/PorterStemmer/def.txt
- Word-Emotion association - https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
- Scikit-Learn - https://scikit-learn.org/stable/


```python
#pip install pandas beautifulsoup4==4.12.3 nltk==3.8.1 wordcloud==1.9.4 matplotlib==3.8.4 nrclex numpy==1.26.4 scikit-learn==1.4.2 
```


```python
import pandas as pd #data processing
from bs4 import BeautifulSoup #HTML tag removing
import re #regular expressions for special characters handling
import nltk
from nltk.tokenize import word_tokenize #tokenization using NLTK library
from nltk.corpus import stopwords #stopwords dictionary
from nltk.stem.porter import PorterStemmer #stemming algorithm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #numerical representation of text
from wordcloud import WordCloud #Graphical representation of top-word in reviews
import matplotlib.pyplot as plt #plots
from nltk.corpus import opinion_lexicon #dictionary of positive and negative words provided by Standford university
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module='nltk') #to avoid warnings 
import numpy as np
from nrclex import NRCLex #Word Emotion Association Lexicon
from sklearn.linear_model import LogisticRegression #logistic regression model
from sklearn.metrics import classification_report #precision, recall
from sklearn.metrics import confusion_matrix #confusion matrix
from sklearn.svm import SVC #Support Vector Classifier
```

Function to predict the sentiment of the review given the model. 


```python
def preprocess_predict(rev, model, pattern=r"[^a-zA-Z\s']"):
    # Preprocess the text
    rev = BeautifulSoup(rev).get_text() #HMTL
    rev = re.sub(pattern, '', rev) #special characters
    rev = word_tokenize(rev) #tokenization
    rev = [word.lower() for word in rev if word.lower() not in stopwords_list] #stopwords
    rev = [stemmer.stem(word) for word in rev] #stemmization 
    processed_rev = ' '.join(rev) #single string rather than separate words

    processed_rev = tfidf.transform([processed_rev]) #TF-IDF vectorization
 
    # Predict sentiment
    prediction = model.predict(processed_rev)
    return prediction[0]
```

Function to assign sentiment score for a tokenized review.


```python
def assign_sent_scores(review, positive_words, negative_words):
    pos_count = sum(1 for word in review if word in positive_words) #sum of positive words
    neg_count = sum(1 for word in review if word in negative_words) #sum of negative words
    return pos_count - neg_count 
```

Function to analyze emotional sentiment of a review. 


```python
def get_emotion_scores(review):
    emotion = NRCLex(review) #initializing the emotion analyis
    
    emotion_scores = emotion.raw_emotion_scores #dictionary of emotions

    ##obtain emotion scores
    emotion_results = {'anger': emotion_scores.get('anger', 0),
                       'anticipation': emotion_scores.get('anticipation', 0),
                       'disgust': emotion_scores.get('disgust', 0),
                        'fear': emotion_scores.get('fear', 0),
                        'joy': emotion_scores.get('joy', 0),
                        'sadness': emotion_scores.get('sadness', 0),
                        'surprise': emotion_scores.get('surprise', 0),
                        'trust': emotion_scores.get('trust', 0),
                        'negative': emotion_scores.get('negative', 0),
                        'positive': emotion_scores.get('positive', 0),
    }
    return emotion_results
```

# Data set characteristics

Large Movie Review Dataset source - https://ai.stanford.edu/~amaas/data/sentiment/

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).


```python
movie_data=pd.read_csv('IMDB Dataset.csv') 
```


```python
print(f'Dataset shape: ', movie_data.shape)
```

    Dataset shape:  (50000, 2)
    

50 000 reviews labelled as either positive or negative

5 first reviews


```python
print(movie_data.head())
```

                                                  review sentiment
    0  One of the other reviewers has mentioned that ...  positive
    1  A wonderful little production. <br /><br />The...  positive
    2  I thought this was a wonderful way to spend ti...  positive
    3  Basically there's a family where a little boy ...  negative
    4  Petter Mattei's "Love in the Time of Money" is...  positive
    


```python
print("Missing values:\n", movie_data.isnull().sum())
```

    Missing values:
     review       0
    sentiment    0
    dtype: int64
    

### Column characteristics


```python
print(f'Number of non-unique reviews: ', movie_data['review'].nunique())
```

    Number of non-unique reviews:  49582
    

Reviews are not unique. It may happen due to single-word review. Some of them are duplicated intentionally by author to balance the target variable.


```python
non_unique = movie_data.groupby('review').filter(lambda x: len(x) > 1)
print(non_unique.sort_values('review'))
```

                                                      review sentiment
    34058  "Go Fish" garnered Rose Troche rightly or wron...  negative
    47467  "Go Fish" garnered Rose Troche rightly or wron...  negative
    29956  "Three" is a seriously dumb shipwreck movie. M...  negative
    31488  "Three" is a seriously dumb shipwreck movie. M...  negative
    47527  "Witchery" might just be the most incoherent a...  negative
    ...                                                  ...       ...
    47876  this movie sucks. did anyone notice that the e...  negative
    44122  well, the writing was very sloppy, the directi...  negative
    23056  well, the writing was very sloppy, the directi...  negative
    10163  when I first heard about this movie, I noticed...  positive
    15305  when I first heard about this movie, I noticed...  positive
    
    [824 rows x 2 columns]
    


```python
movie_data['sentiment'].value_counts()
```




    sentiment
    positive    25000
    negative    25000
    Name: count, dtype: int64



Data is balanced. Duplicates are allowed for balancing the data. 

### HTML tags


```python
movie_data['review'][0]
```




    "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."



Reviews may contain HTML tags like \<br>, therefore text HTML pre-processing is needed. If review does not have any HTML elements, warning may appear.


```python
movie_data['clean_review'] = movie_data['review'].apply(lambda rev: BeautifulSoup(rev).get_text())
```

    C:\Windows\Temp\ipykernel_10328\314088864.py:1: MarkupResemblesLocatorWarning: The input looks more like a filename than markup. You may want to open this file and pass the filehandle into Beautiful Soup.
      movie_data['clean_review'] = movie_data['review'].apply(lambda rev: BeautifulSoup(rev).get_text())
    


```python
#movie_data['clean_review'][0]
```

### Special characters

Special characters and digits like #, $, % should be removed. Only spaces and letters are left in review. The only non-whitespace symbol allowed is apostrophe to be handled by stopword dictionary.


```python
pattern=r"[^a-zA-Z\s']" 
movie_data['clean_review'] = movie_data['clean_review'].apply(lambda rev: re.sub(pattern, '', rev))
```


```python
#movie_data['clean_review'][0]
```

### Text tokenization

Tokenization is the process of breaking down sentences into smaller, more manageable units. The task is to split the review into separate words.


```python
movie_data['clean_token_review'] = movie_data['clean_review'].apply(word_tokenize)
```


```python
#movie_data['clean_token_review'][0]
```

### Stopwords 

For NLP modelling, stopwords do not add much value. By removing them, we focus only on the most important information. Case of the words is ignored to ensure that all of stopwords are correctly removed. 


```python
stopwords_list = stopwords.words('english')
stopwords_list[0:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]




```python
movie_data['clean_token_review'] = movie_data['clean_token_review'].apply(lambda rev: [word.lower() for word in rev if word.lower() not in stopwords_list])
```


```python
#movie_data['clean_token_review'][0]
```

### Stemming

Stemming aims to cut off the ends of words in order to obtain correct root form. Another option would be to use lemmatization - use base, dictionary form of a word. 

We use Porter Stemmer algorithm - set of morphological rules applied to English (e.g. sses --> ss; ies --> i)


```python
stemmer = PorterStemmer()

movie_data['clean_stem_review'] = movie_data['clean_token_review'].apply(lambda rev: [stemmer.stem(word) for word in rev])
```


```python
#movie_data['clean_stem_review'][0]
```

# Empirical analysis (goal, assumptions, results, interpretation)

Goal of the analysis is to correctly predict sentiment of the review using Machine Learning models.


```python
X=movie_data['clean_stem_review']
y=movie_data['sentiment']
```


```python
X = X.astype(str)  
y = y.astype(str)
```

As data is balanced there is no reason to stratify target variable. Number of observations is large enough to use 30% test size. 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3) 
```

### Numerical representation of text

The first approach would have been Count Vectorizer. For each review, it counts the occurrences of each word helping to identify the most popular words for positive and negative reviews. It is simple and intuitive measure, however doesn't considering the importance of words within the entire dataset. 

For this purpose, TF-IDF model is used. 
- Term Frequency (TF) measures how often a word appears in a review, relative to the total words in that dataset.
- Inverse Document Frequency (IDF) measures how common a word is across all reviews. The more reviews a word appears in, the lower its IDF score, reducing its weight. Words that are rare in the dataset get higher IDF scores.

We focus on TF-IDF approach, as we assume that rare words may have greater power to distingiush sentiment.

We fit to the training data only to avoid data leakage.


```python
tfidf = TfidfVectorizer()
train_x_tfidf = tfidf.fit_transform(X_train)
test_x_tfidf = tfidf.transform(X_test)
```

### Top words

Dictionary of words appearing in reviews based on the term frequency and inverse document frequency. 


```python
feature_names = tfidf.get_feature_names_out()
tfidf_scores = train_x_tfidf.sum(axis=0).A1
word_frequencies = dict(zip(feature_names, tfidf_scores))
top_10_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_words 
```




    [('movi', 1843.483371278698),
     ('film', 1544.887345097486),
     ('one', 917.327820975015),
     ('like', 841.5243539376964),
     ('watch', 705.0867676543776),
     ('good', 692.0537966219316),
     ('time', 653.4612625297224),
     ('see', 642.6456435047278),
     ('charact', 621.208675977719),
     ('make', 619.7253827989489)]



Word Cloud is used as graphical representation of the top words in reviews. 


```python
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

# Plot the word cloud.
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud from TF-IDF Scores", fontsize=16)
plt.show()
```


    
![png](output_63_0.png)
    


### Dictionary-based approach 

To analyze sentiment of a review we can use lexicon provided by Standford university. Lists of words are added to GitHub by authors. We label review as positive when score is positive (sum of positive words prevails). 


```python
nltk.download('opinion_lexicon') #download opinion dictionary
nltk.download('punkt')
positive_wds = set(opinion_lexicon.positive()) #https://gist.github.com/mkulakowski2/4289437
negative_wds = set(opinion_lexicon.negative()) #https://gist.github.com/mkulakowski2/4289441
```

    [nltk_data] Downloading package opinion_lexicon to
    [nltk_data]     C:\Users\Ярослав\AppData\Roaming\nltk_data...
    [nltk_data]   Package opinion_lexicon is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\Ярослав\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    

We can test the dictionary on some made up short reviews. 


```python
new_reviews = [
    "The movie was fantastic! I loved it.",
    "It was okay, not the best but not the worst.",
    "The plot was boring and predictable."
]
new_reviews_token = [word_tokenize(review) for review in new_reviews]
```


```python
scores = [assign_sent_scores(review, positive_wds, negative_wds) for review in new_reviews_token]
categories = ["Positive" if score > 0 else "Negative" if score < 0 else "Neutral" for score in scores]
results_nltk = pd.DataFrame({
    "Review": new_reviews,
    "Sentiment Score": scores,
    "Category": categories
})
results_nltk
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment Score</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The movie was fantastic! I loved it.</td>
      <td>2</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>It was okay, not the best but not the worst.</td>
      <td>0</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The plot was boring and predictable.</td>
      <td>-2</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>



Now we can apply the method to our cleaned data. 


```python
scores = [assign_sent_scores(review, positive_wds, negative_wds) for review in movie_data['clean_token_review']]
categories = ["Positive" if score > 0 else "Negative" if score < 0 else "Neutral" for score in scores]
```


```python
results_nltk = pd.DataFrame({
    "Review": movie_data['clean_review'],
    "Sentiment Score": scores,
    "Category": categories
})
results_nltk
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment Score</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>-7</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production The filming tech...</td>
      <td>10</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>5</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>-4</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's Love in the Time of Money is a...</td>
      <td>13</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>I thought this movie did a down right good job...</td>
      <td>13</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>Bad plot bad dialogue bad acting idiotic direc...</td>
      <td>-9</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>I am a Catholic taught in parochial elementary...</td>
      <td>-6</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>I'm going to have to disagree with the previou...</td>
      <td>-7</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>No one expects the Star Trek movies to be high...</td>
      <td>1</td>
      <td>Positive</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 3 columns</p>
</div>




```python
results_nltk['Category'].value_counts()
```




    Category
    Positive    25597
    Negative    21129
    Neutral      3274
    Name: count, dtype: int64



### Dictionary-based approach for sentiment analysis with emotions

For the purpose of sentiment analysis with 8 basic emotion assosiation (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) we use dictionary of National Research Council Canada. 

First we test on sample data.


```python
emot_reviews = ["The movie was absolutely thrilling! I loved every second of it, especially the plot twist at the end. The actors did a fantastic job and the direction was superb. Can't wait to see the next one!",
                 "This movie was a total waste of time. The acting was terrible, the plot was predictable, and the pacing was so slow. I regret spending money on this.",
                "I was really disappointed with the movie. Some scenes were enjoyable, but others were downright disturbing. The special effects were great, though. I expected more from the director."]
```


```python
emotion_data = [get_emotion_scores(review) for review in emot_reviews]
pd.DataFrame(emotion_data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anger</th>
      <th>anticipation</th>
      <th>disgust</th>
      <th>fear</th>
      <th>joy</th>
      <th>sadness</th>
      <th>surprise</th>
      <th>trust</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Same for prepared data.


```python
emotion_data = [get_emotion_scores(review) for review in movie_data['clean_review']]
emotion_results = pd.DataFrame(emotion_data)
emotion_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>anger</th>
      <th>anticipation</th>
      <th>disgust</th>
      <th>fear</th>
      <th>joy</th>
      <th>sadness</th>
      <th>surprise</th>
      <th>trust</th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>6</td>
      <td>3</td>
      <td>13</td>
      <td>2</td>
      <td>11</td>
      <td>1</td>
      <td>9</td>
      <td>20</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>2</td>
      <td>14</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>6</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>13</td>
      <td>10</td>
      <td>14</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>7</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>10</td>
      <td>9</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 10 columns</p>
</div>




```python
emotion_sums = emotion_results.sum() #sum of emotions scores
plt.figure(figsize=(10, 6))
plt.bar(emotion_sums.index, emotion_sums.values, color=plt.cm.rainbow(np.linspace(0, 1, len(emotion_sums))))

# Customize the plot
plt.xlabel('Emotions')
plt.ylabel('Sentiment Scores')
plt.title('Sentiment Scores of Movie Rewiews')
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better visibility
plt.tight_layout()
```


    
![png](output_81_0.png)
    


### Basic logistic regression

In our project we use traditional Machine Learning models - Logistic Regression and Support Vector Machines. Naive Bayesian Classifier is effective for text classification but will be discussed on the lectures later. Neural networks like Recurrent Neural Networks (RNNs) are not used as number of observations is not sufficient for neural network to be efficient in terms of computation terms. 


```python
log_reg = LogisticRegression()
log_reg.fit(train_x_tfidf,y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression()</pre></div> </div></div></div></div>




```python
print(confusion_matrix(y_test,
                           log_reg.predict(test_x_tfidf),
                           labels = ['positive', 'negative']))
```

    [[6828  761]
     [ 907 6504]]
    

761 cases where the model incorrectly predicted review as "negative" when the actual review was "positive"

907 cases where predicted was "positive" when in fact it is "negative"


```python
print(classification_report(y_test,
                            log_reg.predict(test_x_tfidf),
                            labels = ['positive','negative']))
```

                  precision    recall  f1-score   support
    
        positive       0.88      0.90      0.89      7589
        negative       0.90      0.88      0.89      7411
    
        accuracy                           0.89     15000
       macro avg       0.89      0.89      0.89     15000
    weighted avg       0.89      0.89      0.89     15000
    
    

### Support Vector Classifier

The goal of an SVM classifier is to find a decision boundary that best separates the data points of different classes. 


```python
svc = SVC()
svc.fit(train_x_tfidf, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;SVC<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.svm.SVC.html">?<span>Documentation for SVC</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>SVC()</pre></div> </div></div></div></div>




```python
print(confusion_matrix(y_test,
                           svc.predict(test_x_tfidf),
                           labels = ['positive', 'negative']))
```

    [[6903  686]
     [ 882 6529]]
    


```python
print(classification_report(y_test,
                            svc.predict(test_x_tfidf),
                            labels = ['positive','negative']))
```

                  precision    recall  f1-score   support
    
        positive       0.89      0.91      0.90      7589
        negative       0.90      0.88      0.89      7411
    
        accuracy                           0.90     15000
       macro avg       0.90      0.90      0.90     15000
    weighted avg       0.90      0.90      0.90     15000
    
    

### Predict sentiment of a new review with SVC model

Using the function described above we can predict the sentiment with ML model


```python
# Example of model usage
new_review = "An absolutely fantastic film! The actors gave such powerful performances. Highly recommend it!"
sentiment = preprocess_predict(new_review, svc)
print(f"Sentiment: {sentiment}")
```

    Sentiment: positive
    


```python
new_review = "I had high hopes, but this movie fell flat. The story made no sense. Definitely not worth watching."
sentiment = preprocess_predict(new_review, svc)
print(f"Sentiment: {sentiment}")
```

    Sentiment: negative
    


```python
new_review = "The movie had moments of brilliance, especially with the stunning cinematography and the lead actor's heartfelt performance. However, the plot felt unnecessarily convoluted, leaving me confused and disengaged at times. While the music score was mesmerizing and elevated some scenes, the pacing dragged in the second half, making it hard to stay invested. I appreciate the director's ambition, but the execution fell short of the emotional depth it aimed to achieve. It's neither a complete triumph nor a total disaster—just a missed opportunity."
sentiment = preprocess_predict(new_review, log_reg)
print(f"Sentiment: {sentiment}")
```

    Sentiment: negative
    


```python
new_review = "The movie had moments of brilliance, especially with the stunning cinematography and the lead actor's heartfelt performance. However, the plot felt unnecessarily convoluted, leaving me confused and disengaged at times. While the music score was mesmerizing and elevated some scenes, the pacing dragged in the second half, making it hard to stay invested. I appreciate the director's ambition, but the execution fell short of the emotional depth it aimed to achieve. It's neither a complete triumph nor a total disaster—just a missed opportunity."
sentiment = preprocess_predict(new_review, svc)
print(f"Sentiment: {sentiment}")
```

    Sentiment: negative
    
