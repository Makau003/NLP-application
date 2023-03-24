#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import requests
import re
import matplotlib.pyplot as plt
from nltk import FreqDist
from bs4 import BeautifulSoup
from collections import Counter


# In[129]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from wordcloud import WordCloud


# In[20]:


url = "https://www.ippmedia.com/sw/nipashe"


# In[21]:


response = requests.get(url)
html_content = response.text


# In[28]:


soup = BeautifulSoup(html_content, "html.parser")
main_content = soup.find(id = "page")
blog = main_content.text.strip()


# In[29]:


print (blog)


# In[34]:


blog = blog.lower()
blog = re.sub(r'[^\w\s]', '', blog )
blog = re.sub(r'[^\w\s]', '', blog)
blog = re.sub(r"\s+", " ", blog)
blog = re.sub(r'\d+', '', blog)


# In[48]:


print(blog)


# In[56]:


blog_list = [blog]


# In[59]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(blog_list)
y = (url)


# In[62]:


X = X.toarray()


# In[63]:


word_freq = X.sum(axis=0)


# In[64]:


vocab = vectorizer.get_feature_names()


# In[65]:


sorted_idx = word_freq.argsort()[::-1]


# In[66]:


top_words = [vocab[i] for i in sorted_idx[:10]]
top_freqs = [word_freq[i] for i in sorted_idx[:10]]


# In[67]:


plt.bar(top_words, top_freqs)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 10 words in blog posts')
plt.show()


# In[122]:


wordfreq=FreqDist(blog)


# In[124]:


plt.figure(figsize=(10,7.5))
wordfreq.plot(50,cumulative=False)


# In[87]:


text_blob = TextBlob(blog)
sentiment = text_blob.sentiment.polarity


# In[113]:


def analyse_sentiment(tweet):
    analysis=TextBlob(tweet)
    if analysis.sentiment.polarity>0:
        return 'Positive'
    elif analysis.sentiment.polarity==0:
        return 'Neutral'
    else:
        return 'Negative'


# In[114]:


sentence=pd.DataFrame(sentence)


# In[115]:


sentence['sentiment']=[str(analyse_sentiment(x)) for x in sentence.sentence]


# In[116]:


sentiment_counts = sentence.sentiment.value_counts()


# In[117]:


sentence.sentiment.value_counts()


# In[118]:


df = pd.DataFrame({'Sentiment': sentiment_counts.index, 'Count': sentiment_counts.values})


# In[119]:


color_dict = {'positive': 'green', 'negative': 'red', 'neutral': 'gray','other':'blue'}


# In[105]:


ax = df.plot.bar(x='Sentiment', y='Count', rot=0, color=[color_dict.get(sent, color_dict['other']) for sent in df.Sentiment])

ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
ax.set_title('Sentiment Distribution')

plt.show()


# In[126]:


from nltk.tokenize import word_tokenize
blogwords = word_tokenize(blog)
blog_words = " ".join(blogwords)


# In[132]:


wordcloud=WordCloud(width=2000,height=1000).generate(blog)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:




