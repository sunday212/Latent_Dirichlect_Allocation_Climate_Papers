################ Latent Dirichlecht Allocation ################
###################### Topic Modelling ######################

import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import re
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
import nltk
import spacy
from gensim.utils import simple_preprocess
from pprint import pprint
from gensim.models import CoherenceModel
import numpy as np
import pyLDAvis.gensim
import pyLDAvis
import matplotlib.pyplot as plt


#------------------------------------ Data pre-processing ------------------------------
# upload papers
papers = pd.read_csv("/Users/haimannmok/Desktop/Lisa/Careers/SkillingUp/MachineLearning/Mastering_Machine_Learning/Projects/05 LDA/engrxiv-papers.csv", parse_dates = ['Date created', 'Date modified'], dayfirst=True)
papers = papers[['Date created', 'Title', 'Abstract']]

# remove punctuation
papers['Abstract_processed'] = papers['Abstract'].map(lambda x: re.sub("[,\.!?]", "", str(x) ))

# convert to lower case
papers['Abstract_processed'] = papers['Abstract_processed'].map(lambda x: x.lower())

# tokenise words for each row of text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))

data = papers.Abstract_processed.values.tolist()
data_words = list(sent_to_words(data))
print(data_words[:1])

# bi-gram and tri-gram phrase models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stopwords, make bigrams and lemmatize
# nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB','ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags= ['NOUN', 'ADJ', 'VERB', 'ADV'])

# Final processed text
print(data_lemmatized[:1])

# Full list of words
words = [k for sublist in data_lemmatized for k in sublist]


# ---------------------- Explore data ------------------------------

#------- create word cloud -------

strings = ','.join(words).replace(","," ")

wordcloud = WordCloud(background_color = 'white',
                      max_words = 100,
                      contour_width = 3,
                      contour_color = 'blue')
wordcloud.generate(strings)
wordcloud.to_image()
wordcloud.to_file('/Users/haimannmok/environments/lda/word_cloud.png')


#------- Top 10 most popular words -------

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed abstracts
count_data = count_vectorizer.fit_transform(words)

# Visualise the 10 most common words
def plot_10_most_common_words(count_data, count_vectorizer):

    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

plot_10_most_common_words(count_data, count_vectorizer)



# ------------------------------ Build LDA model ------------------------------

#------ Create dictionary & corpus ------

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View Term Document Frequency
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# ------- Find the optimal number of topics -------

# Use Mallet's wrapper for of LDA algorithm
import os
os.environ.update({'MALLET_HOME':r'/Users/haimannmok/environments/lda/mallet-2.0.8/'})
mallet_path = r'/Users/haimannmok/environments/lda/mallet-2.0.8/bin/mallet'

# Iterate number of topics and select model with highest coherence score
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):

        # LDA model
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)

        # Coherence scores
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

# Show graph: number of topics vs coherence score
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.title("Optimal no of topics by max coherence score")
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# Optimal number of topics: 24


#-------- Final model with optimal number of topics --------

num_topics = 14
model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
pprint(model.print_topics(num_words=10))

# Assigned topic labels
# Topic 0: Hydropower
# Topic 1: Engine fuel
# Topic 2: Wave power
# Topic 3: Structure load
# Topic 4: User activity
# Topic 5: Engineering design
# Topic 6: Base model
# Topic 7: Image measurement
# Topic 8: Dynamic flow model
# Topic 9: Network datum
# Topic 10: Sensor technology
# Topic 11: Membrane
# Topic 12: Metal concentrations
# Topic 13: Surface material analysis


#------- Visualize the topics -------

lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(model)
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
LDAvis_data_filepath = os.path.join('./vis'+str(num_topics))

with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(vis, f)

pyLDAvis.save_html(vis, '/Users/haimannmok/environments/lda/vis.html')



#------------------------------ Find the dominant topic for each research paper ------------------------------

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

df_dominant_topic.to_csv("/Users/haimannmok/environments/lda/document_topics.csv", index = False)


# ------------------------------------------- End --------------------------------------------