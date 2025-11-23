import argparse
import os
import re
import string
from dataclasses import dataclass
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Optional plotting
try:
	from matplotlib import pyplot as plt  # type: ignore
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False

try:
	from wordcloud import WordCloud  # type: ignore
	_HAS_WORDCLOUD = True
except Exception:
	_HAS_WORDCLOUD = False

# NLTK setup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk():
	"""
	Download required NLTK datasets if not already available.
	"""
	packages = ["stopwords", "wordnet", "omw-1.4"]
	for pkg in packages:
		try:
			nltk.data.find(f"corpora/{pkg}")
		except LookupError:
			nltk.download(pkg, quiet=True)


_ensure_nltk()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
	"""
	Lowercase, remove punctuation and non-letters, remove stopwords, lemmatize.
	"""
	if not isinstance(text, str):
		return ""
	text = text.lower()
	# Keep letters and spaces
	text = re.sub(r"[^a-z\s]", " ", text)
	# Collapse multiple spaces
	text = re.sub(r"\s+", " ", text).strip()
	words = []
	for token in text.split(" "):
		if not token or token in STOPWORDS:
			continue
		lemma = LEMMATIZER.lemmatize(token)
		if lemma and lemma not in STOPWORDS:
			words.append(lemma)
	return " ".join(words)


@dataclass
class TrainConfig:
	data_csv_path: str
	model_out_path: str
	vectorizer_out_path: str
	test_size: float = 0.2
	random_state: int = 42
	max_features: int = 20000
	min_df: int = 2
	ngram_low: int = 1
	ngram_high: int = 2
	make_plots: bool = False


def load_imdb_csv_or_fallback(csv_path: str) -> Tuple[List[str], List[str]]:
	"""
	Try loading dataset from CSV. If not available, fallback to NLTK movie_reviews.
	CSV is expected to have columns: 'review', 'sentiment' with 'positive'/'negative'.
	"""
	if os.path.exists(csv_path):
		print(f"✓ Found CSV dataset at: {csv_path}")
		df = pd.read_csv(csv_path)
		if "review" not in df.columns or "sentiment" not in df.columns:
			raise ValueError("CSV must contain 'review' and 'sentiment' columns.")
		texts = df["review"].astype(str).tolist()
		labels = df["sentiment"].astype(str).str.lower().tolist()
		print(f"  → Loaded {len(texts)} reviews from CSV dataset")
		pos_count = sum(1 for lbl in labels if lbl == "positive")
		neg_count = sum(1 for lbl in labels if lbl == "negative")
		print(f"  → Positive: {pos_count}, Negative: {neg_count}")
		return texts, labels

	# Fallback to NLTK corpus for a minimal, self-contained demo
	print(f"✗ CSV not found at: {csv_path}")
	print("  → Falling back to NLTK movie_reviews corpus...")
	try:
		nltk.data.find("corpora/movie_reviews")
	except LookupError:
		print("  → Downloading NLTK movie_reviews corpus (first time only)...")
		nltk.download("movie_reviews", quiet=True)
	from nltk.corpus import movie_reviews

	texts = []
	labels = []
	for category in movie_reviews.categories():
		for fid in movie_reviews.fileids(category):
			texts.append(" ".join(movie_reviews.words(fid)))
			labels.append("positive" if category == "pos" else "negative")
	print(f"  → Loaded {len(texts)} reviews from NLTK corpus")
	pos_count = sum(1 for lbl in labels if lbl == "positive")
	neg_count = sum(1 for lbl in labels if lbl == "negative")
	print(f"  → Positive: {pos_count}, Negative: {neg_count}")
	return texts, labels


def vectorize_texts(
	texts: List[str],
	cfg: TrainConfig,
	fit_vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[np.ndarray, TfidfVectorizer]:
	if fit_vectorizer is None:
		vectorizer = TfidfVectorizer(
			max_features=cfg.max_features,
			ngram_range=(cfg.ngram_low, cfg.ngram_high),
			min_df=cfg.min_df,
			analyzer="word"
		)
		X = vectorizer.fit_transform(texts)
		return X, vectorizer
	else:
		X = fit_vectorizer.transform(texts)
		return X, fit_vectorizer


def train_and_evaluate(cfg: TrainConfig) -> None:
	print("Loading dataset...")
	raw_texts, labels = load_imdb_csv_or_fallback(cfg.data_csv_path)

	print("Cleaning texts...")
	cleaned_texts = [clean_text(t) for t in raw_texts]

	print("Vectorizing with TF-IDF...")
	X, vectorizer = vectorize_texts(cleaned_texts, cfg)

	y = np.array([1 if lbl == "positive" else 0 for lbl in labels], dtype=int)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
	)

	print("Training LinearSVC...")
	model = LinearSVC()
	model.fit(X_train, y_train)

	print("Evaluating...")
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc:.4f}")
	print("Classification report:")
	print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

	print(f"Saving model to {cfg.model_out_path}")
	joblib.dump(model, cfg.model_out_path)
	print(f"Saving vectorizer to {cfg.vectorizer_out_path}")
	joblib.dump(vectorizer, cfg.vectorizer_out_path)

	if cfg.make_plots:
		plot_outputs(cfg, cleaned_texts, labels, y_test, y_pred)


def plot_outputs(
	cfg: TrainConfig,
	cleaned_texts: List[str],
	labels: List[str],
	y_test: np.ndarray,
	y_pred: np.ndarray
) -> None:
	if _HAS_MPL:
		cm = confusion_matrix(y_test, y_pred)
		plt.figure(figsize=(4, 4))
		plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
		plt.title("Confusion Matrix")
		plt.colorbar()
		tick_marks = np.arange(2)
		plt.xticks(tick_marks, ["negative", "positive"], rotation=45)
		plt.yticks(tick_marks, ["negative", "positive"])
		thresh = cm.max() / 2.0
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				plt.text(
					j,
					i,
					format(cm[i, j], "d"),
					ha="center",
					va="center",
					color="white" if cm[i, j] > thresh else "black",
				)
		plt.tight_layout()
		plt.xlabel("Predicted")
		plt.ylabel("True")
		out_path = os.path.join(os.path.dirname(cfg.model_out_path), "confusion_matrix.png")
		plt.savefig(out_path, bbox_inches="tight")
		print(f"Saved confusion matrix to {out_path}")
	else:
		print("matplotlib not installed; skipping confusion matrix plot")

	if _HAS_WORDCLOUD:
		pos_text = " ".join([txt for txt, lbl in zip(cleaned_texts, labels) if lbl == "positive"])
		neg_text = " ".join([txt for txt, lbl in zip(cleaned_texts, labels) if lbl == "negative"])
		for name, text in [("positive", pos_text), ("negative", neg_text)]:
			if not text.strip():
				continue
			wc = WordCloud(width=800, height=400, background_color="white").generate(text)
			plt.figure(figsize=(10, 5))
			plt.imshow(wc, interpolation="bilinear")
			plt.axis("off")
			out_wc = os.path.join(os.path.dirname(cfg.model_out_path), f"wordcloud_{name}.png")
			plt.savefig(out_wc, bbox_inches="tight")
			print(f"Saved wordcloud to {out_wc}")
	else:
		print("wordcloud not installed; skipping word cloud plots")


def parse_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Train SVM sentiment model (MovieMood)")
	parser.add_argument(
		"--csv",
		default=os.path.join(os.path.dirname(__file__), "imdb_dataset.csv"),
		help="Path to IMDb dataset CSV with columns 'review','sentiment'",
	)
	parser.add_argument(
		"--model",
		default=os.path.join(os.path.dirname(__file__), "model.joblib"),
		help="Output path for trained model",
	)
	parser.add_argument(
		"--vectorizer",
		default=os.path.join(os.path.dirname(__file__), "vectorizer.joblib"),
		help="Output path for TF-IDF vectorizer",
	)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	parser.add_argument("--max-features", type=int, default=20000)
	parser.add_argument("--min-df", type=int, default=2)
	parser.add_argument("--ngram-low", type=int, default=1)
	parser.add_argument("--ngram-high", type=int, default=2)
	parser.add_argument("--plots", action="store_true", help="Save confusion matrix and wordclouds")
	args = parser.parse_args()
	return TrainConfig(
		data_csv_path=args.csv,
		model_out_path=args.model,
		vectorizer_out_path=args.vectorizer,
		test_size=args.test_size,
		random_state=args.random_state,
		max_features=args.max_features,
		min_df=args.min_df,
		ngram_low=args.ngram_low,
		ngram_high=args.ngram_high,
		make_plots=args.plots,
	)


def main():
	cfg = parse_args()
	train_and_evaluate(cfg)


if __name__ == "__main__":
	main()


