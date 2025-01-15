# NLP - Recommend YouTube Video System Using HuggingFace Pretrained Model !🎥✨
![chatbox-video](https://github.com/RickyDoan/NLP-Recommendation-YT-video-with-Bert-HuggingFace/blob/main/recommendation%20yt%20vid.gif)

## 🔍 Overview:
This system processes a video title or keyword to recommend similar YouTube videos with:
- **AI-based text embedding** (BERT)
- **Cosine similarity indexing**
- Dynamic thumbnail rendering with **Streamlit**

Whether you're a casual user or a data enthusiast, this project showcases how machine learning can create practical, real-world applications.
## 🚀 Features:
- **Text Cleaning & Processing**: Input is cleaned for stopwords, punctuation, etc.
- **Embedding Generation**: Video titles/keywords are converted into meaningful representations using **BERT**.
- **Similarity Computation**: Using **cosine similarity**, we rank videos based on how closely they match your input.
- **Thumbnail Display**: Displays dynamic video thumbnails fetched via file paths.

Get up to **40 recommendations per query** all displayed conveniently in a browser using Streamlit 💡.
## 🧰 Technologies:
- **Python**: Core programming language
- **Transformers**: For BERT-based embedding generation
- **Streamlit**: For building the user interface
- **scikit-learn**: To compute similarities
- **NLTK**: For text preprocessing
- **Pillow (PIL)**: Image handling

## 🕹 How to Use:
1. Clone this repository:
``` bash
   git clone https://github.com/your_username/your_repo_name.git
```
2. Install dependencies via `requirements.txt`:
``` bash
   pip install -r requirements.txt
```
3. Run the application:
``` bash
   streamlit run main_recommendation.py
```

