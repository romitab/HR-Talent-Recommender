
# 🤖 HR Talent Recommender

This project is an AI-powered recommendation system that helps HR teams identify top employee candidates for internal project needs — based on skills, experience, certifications, and sentiment analysis of peer and manager feedback.

---

## 🚀 Features

- 🔍 Natural-language query interface for HR (e.g., “Python + NLP engineers in Bangalore”)
- ✅ Hybrid search (semantic + structured filters)
- 🧠 Sentiment scoring from peer/manager feedback
- 📊 Evaluated using BERTScore and ROUGE metrics
- 📤 Upload your own dataset (CSV)
- 📍 Optional location-based filtering
- 🎯 Streamlit-powered interface

---

## 🛠 Files in this Repo

| File                  | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `app.py`              | Streamlit UI for running the recommender                |
| `model.py`            | Core recommendation engine (FAISS + embeddings + scoring)|
| `requirements.txt`    | Python dependencies                                      |
| `Diverse_HR_Employee_Dataset_200.csv` | Sample dataset with 200 employees          |

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

## 📥 Example HR Queries

- `Looking for ML Engineers with NLP and chatbot experience`
- `Need Data Scientists certified in Azure with 5+ years`
- `AI Engineers in Bangalore with MLOps background`

---

