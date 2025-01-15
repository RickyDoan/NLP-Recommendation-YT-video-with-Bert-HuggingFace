import os
import streamlit as st
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('stopwords')
nltk.download('punkt')

df = load("artifact/dataframe.joblib")
# base_folder = os.path.join(os.getcwd(), "images")
base_image_folder = ("../images")

def clean_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    word = [word for word in words if word not in stop_words and string.punctuation and word.isalnum()]
    return " ".join(word)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def get_text_embedding(text):
    input_ids = tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True, max_length=216)
    with torch.no_grad():
         outputs = model(input_ids)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.cpu().numpy()


def compute_cosine(embedding, embeddings):
    similarities = cosine_similarity(embedding.reshape(1, -1), np.vstack(embeddings)).flatten()
    return similarities

def recommendation_function(text, df , top_n):
    text_cleaned = clean_text(text)
    embedding = get_text_embedding(text_cleaned)
    embeddings = df['text_embedding'].tolist()
    similarity = compute_cosine(embedding, embeddings)
    df['similarity'] = similarity
    df_sorted = df.sort_values(by='similarity', ascending=False)
    # recommendation = df_sorted[df_sorted['Title'] != text].head(top_n)
    recommendation = df_sorted.head(top_n)
    return recommendation[['Id','Title','Channel','Category','image_name','similarity']]


# Function to construct full image paths dynamically
def construct_image_path(base_folder, channel, image_name):
    """
    Constructs the full path to an image using base folder, channel name, and image file name.
    """
    return os.path.join(base_folder, channel, image_name)


# Function to fetch image from constructed path
def fetch_image(image_path):
    """
    Fetch an image from the local directory. If not found, returns None.
    """
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return None

def main():
    st.title("YouTube Video Recommendation System")

    # Input directly below the heading
    input_video = st.text_input("Enter Video Title/Keyword", value="Sample Video Title")
    num_recommendations = 40  # Fixed to showing 10 recommendations

    # Trigger recommendation
    if st.button("Get Recommendations"):
        if input_video.strip():  # Ensure input is valid
            st.write(f"Generating recommendations for: **{input_video}**")

            # Call the recommendation function (with df from your environment)
            recommendations = recommendation_function(input_video, df, top_n=num_recommendations)

            if not recommendations.empty:
                st.write("### Recommended Videos")

                # Display recommendations in rows of 5 images each
                for i in range(0, len(recommendations), 5):
                    # Create a new row with 5 columns
                    columns = st.columns(5)

                    # Loop through the current slice of 5 recommendations
                    for col, (_, row) in zip(columns, recommendations.iloc[i:i + 5].iterrows()):
                        # Construct the image path dynamically
                        image_path = construct_image_path(
                            base_folder=base_image_folder,
                            channel=row['Channel'],
                            image_name=row['image_name']
                        )

                        # Fetch the image
                        img = fetch_image(image_path)

                        with col:
                            if img:  # Display image if available
                                st.image(img, caption=row['Title'], use_container_width=True)
                            else:  # If image is missing
                                st.write(f"**{row['Title']}** (Image not available)")

                # st.success("Recommendations displayed successfully!")
        #     else:
        #         st.warning("No recommendations were found. Try another video.")
        # else:
        #     st.error("Please enter a valid video title.")




if __name__ == "__main__":
    main()