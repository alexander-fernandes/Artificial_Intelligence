from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import sys

global_sentiment = 0.0

class SentimentAnalyzer():
    def __init__(self, ): 
        self.sid_obj = SentimentIntensityAnalyzer()

    def get_scores(self, sentence):
        
        self.sentiment_dict = self.sid_obj.polarity_scores(sentence)
        self.neg_sent = self.sentiment_dict['neg']
        self.neu_sent = self.sentiment_dict['neu']
        self.pos_sent = self.sentiment_dict['pos']
        self.sent_compound = self.sentiment_dict['compound']
        #print("sentence was rated as ", self.neg_sent*100, "% Negative") 
        #print("sentence was rated as ", self.neu_sent*100, "% Neutral") 
        #print("sentence was rated as ", self.pos_sent*100, "% Positive")
        

def predict_sentiment_score(score):
    
    sentiment_total = "neutral"
    
    if score >= 0.05 : 
        sentiment_total = "positive" 
  
    elif score <= - 0.05 : 
        sentiment_total = "negative"
  
    else: 
        score = "neutral"

    return sentiment_total


def get_sentiment_score(sentence):

    sentiment_total = None
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    #print("Overall sentiment dictionary is : ", sentiment_dict) 
    #print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    #print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    #print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
    #print("Sentence Overall Rated As", end = " ") 
  
    # decide sentiment as positive, negative and neutral 
    sentiment_total = predict_sentiment_score(sentiment_dict['compound'])

    return sentiment_total, sentiment_dict['compound']

sentences = []

tricky_sentences = [
    "Most automated sentiment analysis tools are shit.",
    "VADER sentiment analysis is the shit.",
    "Sentiment analysis has never been good.",
    "Sentiment analysis with VADER has never been this good.",
    "Warren Beatty has never been so entertaining.",
    "I won't say that the movie is astounding and I wouldn't claim that the movie is too banal either.",
    "I like to hate Michael Bay films, but I couldn't fault this one",
    "It's one thing to watch an Uwe Boll film, but another thing entirely to pay for it",
    "The movie was too good",
    "This movie was actually neither that funny, nor super witty.",
    "This movie doesn't care about cleverness, wit or any other kind of intelligent humor.",
    "Those who find ugly meanings in beautiful things are corrupt without being charming.",
    "There are slow and repetitive parts, BUT it has just enough spice to keep it interesting.",
    "The script is not fantastic, but the acting is decent and the cinematography is EXCELLENT!",
    "Roger Dodger is one of the most compelling variations on this theme.",
    "Roger Dodger is one of the least compelling variations on this theme.",
    "Roger Dodger is at least compelling as a variation on the theme.",
    "they fall in love with the product",
    "but then it breaks",
    "usually around the time the 90 day warranty expires",
    "the twin towers collapsed today",
    "However, Mr. Carter solemnly argues, his client carried out the kidnapping\
    under orders and in the ''least offensive way possible.''"
]

paragraph = "It was one of the worst movies I've seen, despite good reviews. \
Unbelievably bad acting!! Poor direction. VERY poor production. \
The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

#lines_list = tokenize.sent_tokenize(paragraph)
sentences.extend(tokenize.sent_tokenize(paragraph))
sentences.extend(tricky_sentences)
print("[+] Training Sentiment Analyzer..")
print("-"*60)

toolbar_width = len(sentences)

# setup toolbar
sys.stdout.write("[%s]" % ("-" * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

for sentence in sentences:
    #time.sleep(0.1) # do real work here
    # update the bar
    sys.stdout.write("#")
    sys.stdout.flush()
    #print(sentence + "sentiment: " + get_sentiment_score(sentence))
    print(get_sentiment_score(sentence))

sys.stdout.write("]\n") # this ends the progress bar

print("-"*60)
print("[+] Sentiment Analyzer trained, tested..")
print("-"*60)
print("Say something.")

while True:
    inpt = input(">> ")
    sent_d, sent_comp = get_sentiment_score(inpt)
    print(sent_d, sent_comp)
    global_sentiment += sent_comp
    print("general mood: " + str(global_sentiment), predict_sentiment_score(global_sentiment))
