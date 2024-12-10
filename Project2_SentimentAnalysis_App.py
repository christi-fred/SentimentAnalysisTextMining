## Sample Dash 
import dash
from dash import dcc, html, Input, Output
from textblob import TextBlob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  Data Collection from steam
import steamreviews
# library for language detection
import langdetect
request_params = dict()
# Reference: https://partner.steamgames.com/doc/store/getreviews
request_params['filter'] = 'all'  # reviews are sorted by helpfulness instead of chronology
request_params['day_range'] = '365'  # focus on reviews which were published during the past four weeks
# app product ID
app_id = 1928870
review_dict, query_count = steamreviews.download_reviews_for_app_id(app_id,
                                                                    chosen_request_params=request_params)

keys_to_extract = ['recommendationid', 'review', 'voted_up', 'votes_up', 'weighted_vote_score', 'steam_purchase']


# In[5]:


#Extracting data from app_id(1928870) using reviews keys from dict
data=[]
for k, v in review_dict.items():
    #new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if k == "reviews":
        for i, j in v.items():
            extracted_data = {key: j[key] for key in keys_to_extract if key in j}
            if extracted_data['review'] != '':
                #if langdetect.detect(extracted_data['review']) == 'en':
                data.append(extracted_data)
                    #print(extracted_data['votes_up'])

# store retrieved data into a dataframe
review_df = pd.DataFrame(data)

# visualization showing # of reviews recommended or not recommended based on positive or negative reviews written

Review_counts= review_df['voted_up'].value_counts()
Review_P_counts = review_df['voted_up'].value_counts(normalize=True) * 100

fig, ax1 = plt.subplots()

# Plot the bar chart
Review_counts.plot(kind='bar', ax=ax1, color='blue', edgecolor='black', position=0, width=0)
ax1.set_xlabel('Recommended or Not Recommended')
ax1.set_ylabel('# of Reviews', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for percentages
ax2 = ax1.twinx()
Review_P_counts.plot(kind='bar', ax=ax2, color='blue', edgecolor='black', position=1, width=0.4)
ax2.set_ylabel('% of Reviews', color='red')
ax2.tick_params(axis='y', labelcolor='red')


# Add percentage labels on top of the bars
for index, value in enumerate(Review_P_counts):
    plt.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

# Show the plot
plt.title('Data Distribution by Recommended and Not Recommended Reviews')
plt.close()

# Save the figure to a bytes buffer
import io 
import base64

buf = io.BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)

# Encode the bytes buffer to a base64 string
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close(fig)

# Retreive most frequest texts:
import re
import nltk
nltk.download('stopwords')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem.porter import PorterStemmer

english_words = set(words.words())

ps = PorterStemmer()
all_stopwords = stopwords.words('english')

def custom_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    return [word for word in tokens if word.lower() in english_words]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = all_stopwords, tokenizer=custom_tokenizer)

def featurenames (data):
    X = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()
    return feature_names
def vectorn(data):
    X = vectorizer.fit_transform(data)
    vector = pd.DataFrame(X.toarray(), columns=featurenames(data))
    return vector
allreviewstext = review_df['review'].values

all_feature_names = featurenames(allreviewstext)
vector= vectorn(allreviewstext)

text = []
count = []
for i in range(len(all_feature_names)):
    text.append(all_feature_names[i])
    count.append(vector[all_feature_names[i]].sum())

def topfeatures (txt, count):
    txt = np.array(text)
    cnt = np.array(count)

    # Sort the array in descending order
    sorted_indices = np.argsort(-cnt)

    # Get the top 15 values
    top5_values = cnt[sorted_indices[:20]]
    top5_features = txt[sorted_indices[:20]]

    print("Top 10 values:", top5_values)
    print("Top 10 features:", top5_features)

    top = np.concatenate((top5_features.reshape(len(top5_features),1), top5_values.reshape(len(top5_values),1)),1)
    topn = pd.DataFrame(top, columns = ['features', 'textcounts'])
    topn = topn.convert_dtypes()

    new_dtypes = {'features': 'string', 'textcounts': 'int32'}
    topn = topn.astype(new_dtypes)
    return topn
       
    
# retreive one sample review
allreviewstext_one = allreviewstext[0]

review_text_eng = custom_tokenizer(allreviewstext_one)

# review_text_eng is a list of tokens and stopwords is a set of stopwords
filtered_words = [token for token in review_text_eng if token not in all_stopwords]

from wordcloud import WordCloud

# show word cloud of a single review
wordcloud = WordCloud(
    background_color='white',
    stopwords=all_stopwords,
    max_words=200,
    max_font_size=30,
    scale=3,
    random_state=1)
wordcloud=wordcloud.generate(str(filtered_words))
fig = plt.figure(1, figsize=(30, 30))
plt.title('Word Cloud of a Game Review')
plt.axis('off')

plt.imshow(wordcloud)
buf2 = io.BytesIO()
fig.savefig(buf2, format='png')
buf2.seek(0)

# Encode the bytes buffer to a base64 string
img_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
plt.close(fig)


# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div(children=[
    html.H1(children='Sentiment Analysis App on Game Reviews'),    html.Img(src='https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWs4MzZ0ZDlhbGdnZ2NrandxbGZ2aHY4ZXRqMTB4YjRwZHkxYXRnYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/SMgXv4h6dSc5Q2Ea50/giphy.webp', style={'width': '40%', 'height': 200}),
    html.Div(children='''
        Enter your Review about Minecraft legends:
    '''),
    # Input text area
    dcc.Textarea(
        id='input-text',
        value='',
        style={'width': '40%', 'height': 75}
    ),

    # Button to submit text
    html.Button('Submit', id='submit-button', n_clicks=0),

    # Output sentiment score
    html.Div(id='output-score'),
    html.H2(children='Minecraft Legends Existing Game Reviews Analysis', style = {'position': 'absolute',
            'top': '10px',
            'right': '100px',
            'width': '40%'}),
    html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'position': 'absolute','top': '100px','right': '150px','width': '40%', 'height': '35%'}),
    html.Img(src='data:image/png;base64,{}'.format(img_base64_2), 
    style={'position': 'absolute','top': '350px','right': '150px','width': '40%', 'height': '40%'}
    )
]),

# Define callback to update sentiment score
@app.callback(
    Output('output-score', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        # Get sentiment score
        blob = TextBlob(value)
        sentiment = blob.sentiment
        p_sentiment = sentiment.polarity
        s_sentiment = sentiment.subjectivity
        if p_sentiment/s_sentiment > 0: 
            sentiment_actual = "Positive"
        else:
            sentiment_actual = "Negative"        
        return f'Polarity: {sentiment.polarity:.2f}, Subjectivity: {sentiment.subjectivity:.2f}: Review sentiment is {sentiment_actual}'

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)