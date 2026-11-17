"""
NLP LAB 09: Baseline Feature-Based NER Model
--------------------------------------------
Implements a classical feature-based Named Entity Recognition (NER) baseline
using word-level lexical and sliding window context features paired with
a Naive Bayes / Linear probabilistic classifier.

Evaluated on CoNLL-2003 using strict entity-level metrics via `seqeval`.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB


def parse_conll_file(filepath: str) -> List[Tuple[List[str], List[str]]]:
    """
    Parses a CoNLL-format file (e.g. eng.train, eng.testa, eng.testb).
    Each line contains whitespace-separated columns: [token, pos, chunk, ner].
    Sentences are separated by empty lines.
    Document boundary tokens (-DOCSTART-) are filtered out.
    """
    sentences = []
    current_tokens = []
    current_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens and current_tokens != ["-DOCSTART-"]:
                    sentences.append((current_tokens, current_tags))
                current_tokens, current_tags = [], []
                continue
            parts = line.split()
            token = parts[0]
            tag = parts[-1]
            current_tokens.append(token)
            current_tags.append(tag)

        if current_tokens and current_tokens != ["-DOCSTART-"]:
            sentences.append((current_tokens, current_tags))

    return sentences


def convert_to_iob2(tags: List[str]) -> List[str]:
    """
    Converts CoNLL IOB1 tagging scheme to standard IOB2 (BIO) scheme.
    In IOB1, 'I-' is used even at the start of an entity unless it directly
    follows another entity of the same type. In IOB2, EVERY entity must begin with 'B-'.
    """
    iob2_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            iob2_tags.append("O")
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if i == 0:
                iob2_tags.append("B-" + ent_type)
            else:
                prev_tag = tags[i - 1]
                if prev_tag == "O" or (
                    prev_tag.startswith(("B-", "I-")) and prev_tag[2:] != ent_type
                ):
                    iob2_tags.append("B-" + ent_type)
                else:
                    iob2_tags.append(tag)
        elif tag.startswith("B-"):
            iob2_tags.append(tag)
        else:
            iob2_tags.append(tag)
    return iob2_tags


def word2features(sentence: List[str], i: int) -> Dict[str, object]:
    """
    Extracts handcrafted linguistic, morphological, and local context window features
    for the token at index `i` in `sentence`.
    """
    word = sentence[i]
    features = {
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper": 1 if word.isupper() else 0,
        "word.istitle": 1 if word.istitle() else 0,
        "word.isdigit": 1 if word.isdigit() else 0,
        "word.contains_hyphen": 1 if "-" in word else 0,
    }

    # Preceding token features (window -1)
    if i > 0:
        prev_word = sentence[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle": 1 if prev_word.istitle() else 0,
            "-1:word.isupper": 1 if prev_word.isupper() else 0,
        })
    else:
        features["BOS"] = 1  # Beginning of sentence

    # Subsequent token features (window +1)
    if i < len(sentence) - 1:
        next_word = sentence[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle": 1 if next_word.istitle() else 0,
            "+1:word.isupper": 1 if next_word.isupper() else 0,
        })
    else:
        features["EOS"] = 1  # End of sentence

    return features


def extract_features_and_labels(
    dataset: List[Tuple[List[str], List[str]]]
) -> Tuple[List[Dict[str, object]], List[str]]:
    """
    Flattens sentence-level data into token-level feature dictionaries and labels.
    """
    X_dict = []
    y_labels = []
    for sentence, tags in dataset:
        for i in range(len(sentence)):
            X_dict.append(word2features(sentence, i))
            y_labels.append(tags[i])
    return X_dict, y_labels


def evaluate_baseline(
    clf: MultinomialNB,
    vec: DictVectorizer,
    test_data: List[Tuple[List[str], List[str]]],
) -> Dict[str, float]:
    """
    Predicts labels on test sentences and evaluates with seqeval strict entity metrics.
    """
    X_test_dict, y_test = extract_features_and_labels(test_data)
    X_test_sparse = vec.transform(X_test_dict)
    y_pred_flat = clf.predict(X_test_sparse)

    # Reconstruct sentence-level tag sequences
    true_sentences = []
    pred_sentences = []
    idx = 0
    for sentence, tags in test_data:
        sent_len = len(sentence)
        pred_sentences.append(list(y_pred_flat[idx : idx + sent_len]))
        true_sentences.append(tags)
        idx += sent_len

    strict_f1 = f1_score(true_sentences, pred_sentences)
    strict_prec = precision_score(true_sentences, pred_sentences)
    strict_rec = recall_score(true_sentences, pred_sentences)
    report = classification_report(true_sentences, pred_sentences, digits=4)

    return {
        "f1": strict_f1,
        "precision": strict_prec,
        "recall": strict_rec,
        "report": report,
        "true_sentences": true_sentences,
        "pred_sentences": pred_sentences,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Classical Word-Feature Baseline NER on CoNLL-2003"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="d:/My_project/NLP_LABS/lab_09/data/eng.train",
        help="Path to training data",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="d:/My_project/NLP_LABS/lab_09/data/eng.testb",
        help="Path to test data",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Laplace smoothing parameter for Naive Bayes",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("NLP LAB 09: CLASSICAL WORD-FEATURE NER BASELINE")
    print("=" * 70)
    print(f"Loading training data from: {args.train_file}")
    print(f"Loading test data from:     {args.test_file}")

    train_raw = parse_conll_file(args.train_file)
    test_raw = parse_conll_file(args.test_file)

    train_data = [(tokens, convert_to_iob2(tags)) for tokens, tags in train_raw]
    test_data = [(tokens, convert_to_iob2(tags)) for tokens, tags in test_raw]

    print(f"Loaded {len(train_data)} training sentences, {len(test_data)} test sentences.")

    print("\nExtracting handcrafted morphological and context window features...")
    t_feat_start = time.time()
    X_train_dict, y_train = extract_features_and_labels(train_data)
    feat_time = time.time() - t_feat_start
    print(f"Feature extraction completed in {feat_time:.2f}s ({len(y_train)} tokens).")

    print("Vectorizing feature dictionaries into sparse matrix...")
    vec = DictVectorizer()
    X_train_sparse = vec.fit_transform(X_train_dict)
    print(f"Vocabulary feature dimension: {X_train_sparse.shape[1]:,} features.")

    print(f"Training Multinomial Naive Bayes classifier (alpha={args.alpha})...")
    t_train_start = time.time()
    clf = MultinomialNB(alpha=args.alpha)
    clf.fit(X_train_sparse, y_train)
    train_time = time.time() - t_train_start
    print(f"Model training completed in {train_time:.2f}s.")

    print("\nEvaluating on Test Set (Strict Entity-Level via seqeval)...")
    results = evaluate_baseline(clf, vec, test_data)

    print("\n" + "=" * 70)
    print("BASELINE EVALUATION REPORT (Strict Span Match)")
    print("=" * 70)
    print(results["report"])
    print(f"Overall Strict Entity F1-Score: {results['f1']:.4f}")
    print(f"Overall Strict Entity Precision: {results['precision']:.4f}")
    print(f"Overall Strict Entity Recall:    {results['recall']:.4f}")
    print(f"Total Model Parameters:          {X_train_sparse.shape[1] * len(clf.classes_):,}")
    print(f"Training Wall-Clock Time:        {train_time:.4f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
