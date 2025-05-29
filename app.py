import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from twitter_client import TwitterClient  # Your module that fetches tweets

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Predict sentiment
def predict_sentiment(text):
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    return model.predict(vect)[0]  # 1 = positive, 0 = negative

# Streamlit UI
st.title("Twitter Sentiment Analysis")
st.write("Analyze the public sentiment on Twitter about any topic!")

topic = st.text_input("Enter a topic (e.g., 'AI', 'Elections', 'Bitcoin')")

if st.button("Analyze"):
    st.info(f"Fetching tweets about: **{topic}**")
    api = TwitterClient()
    raw_tweets = api.get_tweets(query=topic, count=100)

    if not raw_tweets:
        st.warning("No tweets fetched. Try another topic or check API limits.")
    else:
        # Predict sentiment for each tweet
        results = []
        for tweet in raw_tweets:
            text = tweet['text']
            sentiment = predict_sentiment(text)
            results.append({'text': text, 'sentiment': sentiment})

        # Convert to DataFrame
        df = pd.DataFrame(results)
        pos_count = df['sentiment'].sum()
        neg_count = len(df) - pos_count

        # Display percentages
        pos_percent = 100 * pos_count / len(df)
        neg_percent = 100 * neg_count / len(df)

        st.success("Sentiment analysis complete!")
        st.write(f"✅ Positive tweets: **{pos_percent:.2f}%**")
        st.write(f"❌ Negative tweets: **{neg_percent:.2f}%**")

        # Pie chart
        st.subheader("🧁 Sentiment Distribution")
        pie_labels = ['Positive', 'Negative']
        pie_sizes = [pos_count, neg_count]
        fig1, ax1 = plt.subplots()
        ax1.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=["green", "red"])
        ax1.axis('equal')
        st.pyplot(fig1)

        # Bar chart
        st.subheader("📊 Tweet Sentiment Count")
        fig2, ax2 = plt.subplots()
        ax2.bar(pie_labels, pie_sizes, color=["green", "red"])
        ax2.set_ylabel("Number of Tweets")
        st.pyplot(fig2)

        # Show sample tweets
        st.subheader("📈 Sample Positive Tweets")
        for tweet in df[df['sentiment'] == 1]['text'].head(5):
            st.write(f"🟢 {tweet}")

        st.subheader("📉 Sample Negative Tweets")
        for tweet in df[df['sentiment'] == 0]['text'].head(5):
            st.write(f"🔴 {tweet}")
