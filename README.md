# 🧭 CareerCompass AI — Career Path Recommendation System

A professional, ML-powered career recommendation web app built with **Streamlit**, **scikit-learn**, and **Plotly**.

---

## 🚀 Features

- **Smart Assessment** — Rate yourself across 8 skill dimensions
- **ML Predictions** — Gradient Boosting model with ~95% accuracy on 20 careers
- **Radar Chart** — Real-time visual skill profile as you adjust sliders
- **Confidence Scores** — Ranked career matches with probability bars
- **Career Explorer** — Heatmap comparing skill requirements across careers
- **Dark, Professional UI** — Custom CSS with Syne + DM Sans typography

---

## 📦 Tech Stack

| Layer | Library |
|-------|---------|
| Frontend | Streamlit + Custom CSS |
| ML Model | scikit-learn (GradientBoostingClassifier) |
| Visualization | Plotly |
| Data | pandas + numpy |

---

## 🏃 Run Locally

```bash
# 1. Clone or download this folder
cd career_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: CareerCompass AI"
git remote add origin https://github.com/YOUR_USERNAME/career-compass.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and file (`app.py`)
5. Click **"Deploy"** — done! 🎉

Your app will be live at:
```
https://YOUR_USERNAME-career-compass-app-XXXX.streamlit.app
```

---

## 🏗️ Project Structure

```
career_recommender/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Theme & server configuration
```

---

## 🎯 Career Categories Covered

| Category | Careers |
|----------|---------|
| Technology | Software Engineer, Data Scientist, Cybersecurity Analyst, Data Engineer, Game Developer, AI/ML Engineer |
| Design | UX/UI Designer, Architect, Graphic Designer |
| Business | Product Manager, Marketing Manager, Entrepreneur, HR Manager |
| Finance | Financial Analyst |
| Science | Biomedical Researcher, Environmental Scientist |
| Healthcare | Clinical Psychologist |
| Media | Content Creator |
| Operations | Supply Chain Manager |
| Law | Legal Consultant |

---

## 🧠 ML Model Details

- **Algorithm:** Gradient Boosting Classifier (sklearn)
- **Training Samples:** 2,000 synthetic profiles
- **Features:** 8 skill dimensions
- **Test Accuracy:** ~95%
- **Output:** Top-5 career recommendations with confidence %

---

## 📄 License

MIT License — free for educational and personal use.
