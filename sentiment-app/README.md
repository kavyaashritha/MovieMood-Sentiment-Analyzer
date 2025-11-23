## 🎬 MovieMood — Sentiment Analyzer

Full-stack web app that predicts whether an IMDb-like movie review is Positive or Negative.

- Frontend: React (Vite) + Tailwind CSS + Framer Motion
- Backend: FastAPI (Python)
- ML: scikit-learn SVM (`LinearSVC`) + TF-IDF
- Preprocessing: lowercase, regex cleanup, stopwords removal, lemmatization (NLTK)

The backend can train from a local `imdb_dataset.csv` (Kaggle) or automatically fall back to the NLTK `movie_reviews` corpus for a quick demo.


### Folder Structure

```
sentiment-app/
├── backend/
│   ├── app.py
│   ├── model_training.py
│   ├── imdb_dataset.csv           # optional: if you have Kaggle CSV
│   ├── model.joblib               # generated after training
│   ├── vectorizer.joblib          # generated after training
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       ├── index.css
│       ├── main.jsx
│       └── components/
│           ├── InputBox.jsx
│           └── ResultCard.jsx
└── README.md
```


## 1) Backend Setup (Windows-friendly)

Prereqs:
- Python 3.11+ recommended

Steps (PowerShell):

```powershell
cd "sentiment-app\backend"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you have the Kaggle IMDb dataset, place it at `sentiment-app/backend/imdb_dataset.csv` with columns:

```csv
review,sentiment
This movie was great!,positive
I disliked the plot.,negative
```

Train the model (will also download necessary NLTK resources). If `imdb_dataset.csv` is missing, it will fall back to the NLTK movie_reviews corpus:

```powershell
python model_training.py
```

Optional: save plots (confusion matrix, word clouds):

```powershell
python model_training.py --plots
```

Run the API:

```powershell
python app.py
```

The FastAPI server starts at `http://127.0.0.1:8000`.

Test quickly:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Body (@{ review="Amazing acting and story!" } | ConvertTo-Json) -ContentType "application/json"
```

Expected response:

```json
{ "prediction": "positive" }
```


## 2) Frontend Setup

Prereqs:
- Node.js 18+ and npm

Steps:

```powershell
cd "sentiment-app\frontend"
npm install
npm run dev
```

Open the app in your browser (Vite dev server):

- `http://localhost:5173`

If your backend is on a different URL, set an environment variable for the frontend:

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run dev
```


## 3) Usage

1. Start backend: `python app.py` (port 8000).
2. Start frontend: `npm run dev` (port 5173).
3. Enter/paste a movie review and click “Analyze Sentiment”.
4. The result card shows Positive (green) or Negative (red).


## Notes

- The backend uses NLTK stopwords and WordNet lemmatizer. These are auto-downloaded on first run.
- If `model.joblib` and `vectorizer.joblib` are not found when starting the API, the backend will train a model automatically (from CSV if present, otherwise from the NLTK corpus).
- CORS is enabled for `http://localhost:5173` (and `*` for convenience in demo). Tighten before production.


## Troubleshooting

- If NLTK downloads are blocked by corporate network, download manually and place in your NLTK data directory.
- If `wordcloud` or `matplotlib` are not available, training still works, but plots are skipped.
- If frontend cannot reach backend, ensure the ports match and firewall allows 8000.


## License

MIT
*** End Patch``` }კითხ ***!

