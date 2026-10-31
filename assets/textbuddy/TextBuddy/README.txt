TextBuddy — Python companion

Setup (Python 3):
  python3 -m pip install -r requirements.txt
  python3 -m nltk.downloader stopwords averaged_perceptron_tagger_eng
  python3 TextBuddy.py

Choose 4 for the built-in short story, or 1–3 for files, URL or a folder.
The Python version generates Word/Excel reports and word clouds.

This companion preserves the corrected POS-based LD implementation. Its TTR
tokenizer is aligned with the website: lowercase English words, internal
apostrophes preserved, hyphenated parts split, numbers excluded. Python
word-frequency analysis uses NLTK stopwords plus the original custom list;
the browser uses a compact bundled stopword list, so frequencies may differ.
LD uses a separate Treebank/POS tokenization (including split contractions).
POS tags and auxiliary identification are approximate; LD is not readability.

Sample: The Fox and the Cat, complete English text from Grimms' Fairy Tales,
by Jacob and Wilhelm Grimm. The Gutenberg text is based on translations by
Edgar Taylor and Marian Edwardes. No story wording was abridged or rewritten.
Source: https://www.gutenberg.org/files/2591/2591-h/2591-h.htm
Catalogue: https://www.gutenberg.org/ebooks/2591
Sample sentiment was calculated using vaderSentiment 3.3.2: compound 0.844.
The sample-analysis.json contains no estimated LD value.
