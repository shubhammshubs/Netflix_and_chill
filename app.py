import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("netflix_titles.csv")  # Replace with actual file

# Preprocess data
df['content'] = df['listed_in'] * 3 + " " + df['description']
df.dropna(subset=['title', 'content'], inplace=True)

# Convert text to numerical data
vectorizer = CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(df['content'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Recommendation function
def recommend(title):
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return ["Title not found! Try another one."]
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    return df.iloc[[i[0] for i in sim_scores]]['title'].tolist()

# Streamlit UI
st.title("🎬 Netflix Movie/Show Recommender")
st.write("Enter a movie/show name to get recommendations.")

title = st.text_input("Enter a movie/show title:")
if title:
    recommendations = recommend(title)
    st.subheader("Recommended for you:")
    for rec in recommendations:
        st.write(f"- {rec}")
