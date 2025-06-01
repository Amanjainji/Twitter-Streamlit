import re
import joblib
import tweepy
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')  # Ensure it's available

class TwitterClient(object):
    '''
    Twitter Client for sentiment analysis using custom ML model.
    '''
    def __init__(self):
        # Initialize Twitter API v2 Client with your Bearer Token
        self.client = tweepy.Client(bearer_token='Your Bearer Token')

        # Load trained sentiment analysis model and vectorizer
        self.model = joblib.load('sentiment_model.pkl')
        self.vectorizer = joblib.load('tfidf_vectorizer.pkl')

        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_tweet(self, tweet):
        '''
        Clean and stem tweet text.
        '''
        # Remove mentions, URLs, and special characters
        tweet = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet)
        words = tweet.lower().split()
        stemmed = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(stemmed)

    def get_tweet_sentiment(self, tweet):
        '''
        Predict sentiment using trained model.
        '''
        cleaned = self.clean_tweet(tweet)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        return 'positive' if prediction == 1 else 'negative'

    def get_tweets(self, query, count=100):
        '''
        Fetch and process tweets using Twitter API v2.
        '''
        tweets = []
        try:
            response = self.client.search_recent_tweets(
                query=f"{query} lang:en -is:retweet",
                max_results=min(count, 100),
                tweet_fields=['text']
            )
            if response.data:
                for tweet in response.data:
                    parsed_tweet = {
                        'text': tweet.text,
                        'sentiment': self.get_tweet_sentiment(tweet.text)
                    }
                    tweets.append(parsed_tweet)
            return tweets
        except Exception as e:
            print("Error:", e)
            return []

def main():
    api = TwitterClient()
    tweets = api.get_tweets(query='Donald Trump', count=100)

    ptweets = [t for t in tweets if t['sentiment'] == 'positive']
    ntweets = [t for t in tweets if t['sentiment'] == 'negative']

    if len(tweets) != 0:
        print("Sentiment analysis done")
        print("Positive tweets percentage: {:.2f} %".format(100 * len(ptweets) / len(tweets)))
        print("Negative tweets percentage: {:.2f} %".format(100 * len(ntweets) / len(tweets)))

        print("\n\nPositive tweets:")
        for tweet in ptweets[:10]:
            print(tweet['text'])

        print("\n\nNegative tweets:")
        for tweet in ntweets[:10]:
            print(tweet['text'])
    else:
        print("No tweets fetched or API limit issue.")

if __name__ == "__main__":
    main()
