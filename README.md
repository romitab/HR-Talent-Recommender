
# 🙆 HR Talent Recommender

This project is an AI-powered recommendation system that helps HR teams identify top employee candidates for internal project needs — based on skills, experience, certifications, and sentiment analysis of peer and manager feedback using a **hybrid recommendation model** with semantic search, structured filtering, and sentiment analysis.

---

## 🚀 Features

- 🔍 Natural-language query interface for HR (e.g., “Python + NLP engineers in Bangalore”)
- ✅ Hybrid search (semantic + structured filters)
- 🧠 Sentiment scoring from peer/manager feedback
- 📊 Evaluated using BERTScore and ROUGE metrics
- 📍 Optional location-based filtering
- 🔢 Dynamic `top_k` selection
- 📤 Supports **dataset updating** via file upload
- 🌐 Streamlit-powered interface

---

## 🗂 Files in this Repo

| File                               | Purpose                                                  |
|------------------------------------|----------------------------------------------------------|
| `app.py`                           | Streamlit UI for running the recommender                |
| `model.py`                         | Core recommendation engine (FAISS + embeddings + scoring)|
| `requirements.txt`                 | Python dependencies                                      |
| `Diverse_HR_Employee_Dataset_200.csv` | Sample dataset with 200 employees                    |
| `local_model/`                     | Pre-downloaded model for offline use on Streamlit Cloud |

---

## 💻 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Fork or clone this repo
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub, select this repo
4. Choose `app.py` as the main file and deploy

---

## 🔄 Why `local_model/`?

Streamlit Cloud **does not allow runtime model downloads**.  
To prevent errors like `OSError` or missing model files, this project includes a pre-saved model folder called `local_model/`.

It contains a local copy of:
```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Usage in `model.py`:
```python
embedder = SentenceTransformer("local_model")
```

✅ This allows the app to run **fully offline and error-free** in Streamlit Cloud or containerized environments.

---

## 📝 Example HR Queries

- `Looking for ML Engineers with NLP and chatbot experience`
- `Need Data Scientists certified in Azure with 5+ years`
- `AI Engineers in Bangalore with MLOps background`

---

Made with ❤️ for smart HR teams.
