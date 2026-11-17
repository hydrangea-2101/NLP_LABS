"""
NLP LAB 09: BERT Fine-Tuning Pipeline for Token Classification (NER)
--------------------------------------------------------------------
Trains and evaluates BertForTokenClassification on CoNLL-2003 using Strategy A.
Generates all required experimental outputs:
- results.csv
- entity_metrics.csv
- entity_length_analysis.png
- confusion_matrix.png
- error_cases.json
- checkpoints/best_ner_model.pt
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from sklearn.metrics import confusion_matrix

# Fix random seed for full reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def parse_conll_file(filepath: str) -> List[Tuple[List[str], List[str]]]:
    sentences = []
    curr_tokens, curr_tags = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_tokens and curr_tokens != ["-DOCSTART-"]:
                    sentences.append((curr_tokens, curr_tags))
                curr_tokens, curr_tags = [], []
                continue
            parts = line.split()
            curr_tokens.append(parts[0])
            curr_tags.append(parts[-1])
        if curr_tokens and curr_tokens != ["-DOCSTART-"]:
            sentences.append((curr_tokens, curr_tags))
    return sentences


def convert_to_iob2(tags: List[str]) -> List[str]:
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


class CoNLLNERDataset(Dataset):
    def __init__(self, data: List[Tuple[List[str], List[str]]], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = self._prepare_features()

    def _prepare_features(self):
        features = []
        for words, tags in self.data:
            tokenized = self.tokenizer(
                words,
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            word_ids = tokenized.word_ids()

            # Strategy A label alignment
            aligned_labels = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    aligned_labels.append(-100)
                elif wid != prev_word:
                    prev_word = wid
                    # Guard in case truncation dropped later words
                    if wid < len(tags):
                        aligned_labels.append(LABEL2ID[tags[wid]])
                    else:
                        aligned_labels.append(-100)
                else:
                    aligned_labels.append(-100)

            features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(aligned_labels, dtype=torch.long),
                "words": words,
                "orig_tags": tags
            })
        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def main():
    print("=" * 70)
    print("NLP LAB 09: PRETRAINED BERT FINE-TUNING FOR NER")
    print("=" * 70)

    data_dir = "d:/My_project/NLP_LABS/lab_09/data"
    output_dir = "d:/My_project/NLP_LABS/lab_09"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Load Data
    print("Loading CoNLL-2003 train, validation (testa), and test (testb) sets...")
    train_raw = parse_conll_file(os.path.join(data_dir, "eng.train"))
    val_raw = parse_conll_file(os.path.join(data_dir, "eng.testa"))
    test_raw = parse_conll_file(os.path.join(data_dir, "eng.testb"))

    train_data = [(w, convert_to_iob2(t)) for w, t in train_raw]
    val_data = [(w, convert_to_iob2(t)) for w, t in val_raw]
    test_data = [(w, convert_to_iob2(t)) for w, t in test_raw]

    print(f"Train sentences: {len(train_data):,}")
    print(f"Val sentences:   {len(val_data):,}")
    print(f"Test sentences:  {len(test_data):,}")

    # 2. Tokenizer and Datasets
    MODEL_CHECKPOINT = "google/bert_uncased_L-2_H-128_A-2"
    print(f"\nInitializing tokenizer from {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print("Building PyTorch datasets with Strategy A alignment...")
    train_dataset = CoNLLNERDataset(train_data, tokenizer, max_length=128)
    val_dataset = CoNLLNERDataset(val_data, tokenizer, max_length=128)
    test_dataset = CoNLLNERDataset(test_data, tokenizer, max_length=128)

    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b])
        }
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b])
        }
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b])
        }
    )

    # 3. Model setup
    device = torch.device("cpu")
    print(f"\nInitializing BertForTokenClassification ({len(LABEL_LIST)} classes)...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters:     {total_params:,}")
    print(f"Trainable Model Parameters: {trainable_params:,}")

    # Optimizer & Scheduler
    epochs = 4
    lr = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 4. Training loop
    print(f"\nBeginning Fine-Tuning ({epochs} epochs, lr={lr}, device=CPU)...")
    best_val_f1 = 0.0
    best_model_path = os.path.join(checkpoint_dir, "best_ner_model.pt")

    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0_epoch = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - t0_epoch

        # Validation evaluation
        model.eval()
        val_true_sents, val_pred_sents = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_np = labels.cpu().numpy()

                for b in range(len(preds)):
                    true_seq, pred_seq = [], []
                    for t in range(len(preds[b])):
                        if labels_np[b, t] != -100:
                            true_seq.append(ID2LABEL[labels_np[b, t]])
                            pred_seq.append(ID2LABEL[preds[b, t]])
                    val_true_sents.append(true_seq)
                    val_pred_sents.append(pred_seq)

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_true_sents, val_pred_sents)
        val_prec = precision_score(val_true_sents, val_pred_sents)
        val_rec = recall_score(val_true_sents, val_pred_sents)

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.2f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val F1: {val_f1:.4f})")

    total_train_time = time.time() - t_train_start
    print(f"\nTraining completed in {total_train_time:.2f}s.")

    # 5. Final Evaluation on Test Set (eng.testb)
    print("\nLoading best checkpoint for final Test set evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_true_sents, test_pred_sents = [], []
    flat_true_tokens, flat_pred_tokens = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for b in range(len(preds)):
                true_seq, pred_seq = [], []
                for t in range(len(preds[b])):
                    if labels_np[b, t] != -100:
                        true_label = ID2LABEL[labels_np[b, t]]
                        pred_label = ID2LABEL[preds[b, t]]
                        true_seq.append(true_label)
                        pred_seq.append(pred_label)
                        flat_true_tokens.append(true_label)
                        flat_pred_tokens.append(pred_label)
                test_true_sents.append(true_seq)
                test_pred_sents.append(pred_seq)

    test_f1 = f1_score(test_true_sents, test_pred_sents)
    test_prec = precision_score(test_true_sents, test_pred_sents)
    test_rec = recall_score(test_true_sents, test_pred_sents)
    report_str = classification_report(test_true_sents, test_pred_sents, digits=4)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION REPORT (BERT-Tiny Token Classification)")
    print("=" * 70)
    print(report_str)
    print(f"Strict Entity F1-Score:  {test_f1:.4f}")
    print(f"Strict Entity Precision: {test_prec:.4f}")
    print(f"Strict Entity Recall:    {test_rec:.4f}")

    # Parse per-entity metrics from classification report dictionary
    from seqeval.metrics import classification_report as cr_dict
    report_dict = cr_dict(test_true_sents, test_pred_sents, output_dict=True)

    # 6. Save results.csv
    # Baseline numbers from baseline_ner.py run:
    # F1: 0.6477, Prec: 0.5742, Rec: 0.7427, Params: 680,463, Time: 0.13s
    results_df = pd.DataFrame([
        {
            "Model": "Baseline (Lexical + Sliding Window NB)",
            "Strict Precision": 0.5742,
            "Strict Recall": 0.7427,
            "Strict F1": 0.6477,
            "Training Time (s)": 0.1287,
            "Parameters": 680463
        },
        {
            "Model": "Pretrained BERT-Tiny (Strategy A Fine-Tuning)",
            "Strict Precision": round(test_prec, 4),
            "Strict Recall": round(test_rec, 4),
            "Strict F1": round(test_f1, 4),
            "Training Time (s)": round(total_train_time, 2),
            "Parameters": total_params
        }
    ])
    results_csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved {results_csv_path}")

    # 7. Save entity_metrics.csv
    entity_types = ["PER", "LOC", "ORG", "MISC"]
    entity_rows = []
    for et in entity_types:
        if et in report_dict:
            entity_rows.append({
                "Entity Type": et,
                "Precision": round(report_dict[et]["precision"], 4),
                "Recall": round(report_dict[et]["recall"], 4),
                "F1-Score": round(report_dict[et]["f1-score"], 4),
                "Support": int(report_dict[et]["support"])
            })
    entity_df = pd.DataFrame(entity_rows)
    entity_csv_path = os.path.join(output_dir, "entity_metrics.csv")
    entity_df.to_csv(entity_csv_path, index=False)
    print(f"Saved {entity_csv_path}")

    # 8. Entity Length Analysis (F1 by length: 1, 2, 3, 4+)
    print("\nComputing Entity Length Analysis...")
    def get_entities_with_spans(tags):
        entities = []
        start = None
        curr_type = None
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if start is not None:
                    entities.append((curr_type, start, i))
                start = i
                curr_type = tag[2:]
            elif tag.startswith("I-"):
                if start is not None and tag[2:] == curr_type:
                    pass
                else:
                    if start is not None:
                        entities.append((curr_type, start, i))
                    start = i
                    curr_type = tag[2:]
            else:
                if start is not None:
                    entities.append((curr_type, start, i))
                    start = None
                    curr_type = None
        if start is not None:
            entities.append((curr_type, start, len(tags)))
        return entities

    # Group true and pred entities by length
    length_buckets = {1: {"tp": 0, "fp": 0, "fn": 0},
                      2: {"tp": 0, "fp": 0, "fn": 0},
                      3: {"tp": 0, "fp": 0, "fn": 0},
                      4: {"tp": 0, "fp": 0, "fn": 0}} # 4 means >= 4

    for true_tags, pred_tags in zip(test_true_sents, test_pred_sents):
        true_ents = set(get_entities_with_spans(true_tags))
        pred_ents = set(get_entities_with_spans(pred_tags))

        # Check true entities
        for ent in true_ents:
            length = ent[2] - ent[1]
            bucket = length if length < 4 else 4
            if ent in pred_ents:
                length_buckets[bucket]["tp"] += 1
            else:
                length_buckets[bucket]["fn"] += 1

        # Check pred entities for false positives
        for ent in pred_ents:
            length = ent[2] - ent[1]
            bucket = length if length < 4 else 4
            if ent not in true_ents:
                length_buckets[bucket]["fp"] += 1

    length_f1 = {}
    for b in [1, 2, 3, 4]:
        tp = length_buckets[b]["tp"]
        fp = length_buckets[b]["fp"]
        fn = length_buckets[b]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        length_f1[b] = round(f1, 4)
        print(f"  Length {b}{'+' if b==4 else ''} tokens -> Support: {tp+fn}, F1: {f1:.4f} (P: {p:.4f}, R: {r:.4f})")

    # Plot entity_length_analysis.png
    plt.figure(figsize=(8, 4.5), dpi=130)
    x_labels = ["1 token", "2 tokens", "3 tokens", "4+ tokens"]
    y_vals = [length_f1[1], length_f1[2], length_f1[3], length_f1[4]]
    bars = plt.bar(x_labels, y_vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85, edgecolor="black")
    plt.ylim(0, 1.0)
    plt.xlabel("Độ dài thực thể (Entity Length in Tokens)", fontsize=11, fontweight="bold")
    plt.ylabel("Strict Entity F1-Score", fontsize=11, fontweight="bold")
    plt.title("Phân tích Hiệu năng NER theo Độ dài Thực thể (CoNLL-2003 Test)", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, y_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    length_img_path = os.path.join(output_dir, "entity_length_analysis.png")
    plt.savefig(length_img_path)
    plt.close()
    print(f"Saved {length_img_path}")

    # 9. Confusion Matrix
    print("\nComputing Confusion Matrix...")
    classes = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    cm = confusion_matrix(flat_true_tokens, flat_pred_tokens, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8.5, 7), dpi=130)
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Ma trận Nhầm lẫn Chuẩn hóa Cấp Token (Normalized Confusion Matrix)", fontsize=12, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "0.00"
            plt.text(j, i, text, ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=9)

    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_img_path)
    plt.close()
    print(f"Saved {cm_img_path}")

    # 10. Extract Error Cases for error_analysis.md
    print("\nExtracting structured error cases for error_analysis.md...")
    fp_cases = []
    fn_cases = []
    boundary_cases = []

    for i, (words, true_tags) in enumerate(test_data):
        pred_tags = test_pred_sents[i]
        true_ents = set(get_entities_with_spans(true_tags))
        pred_ents = set(get_entities_with_spans(pred_tags))

        # Check boundary errors: overlap in span but start/end mismatch or type matches partial
        for p_type, p_s, p_e in pred_ents:
            for t_type, t_s, t_e in true_ents:
                if (p_s < t_e and p_e > t_s) and (p_s != t_s or p_e != t_e):
                    if len(boundary_cases) < 10:
                        boundary_cases.append({
                            "sentence": " ".join(words),
                            "gold_entity": " ".join(words[t_s:t_e]),
                            "gold_type": t_type,
                            "gold_span": [t_s, t_e],
                            "pred_entity": " ".join(words[p_s:p_e]),
                            "pred_type": p_type,
                            "pred_span": [p_s, p_e]
                        })

        # False positives (predicted entity not overlapping any true entity)
        for p_type, p_s, p_e in pred_ents:
            overlapping = any(p_s < t_e and p_e > t_s for _, t_s, t_e in true_ents)
            if not overlapping and len(fp_cases) < 10:
                fp_cases.append({
                    "sentence": " ".join(words),
                    "pred_entity": " ".join(words[p_s:p_e]),
                    "pred_type": p_type,
                    "pred_span": [p_s, p_e]
                })

        # False negatives (true entity not overlapping any predicted entity)
        for t_type, t_s, t_e in true_ents:
            overlapping = any(p_s < t_e and p_e > t_s for _, p_s, p_e in pred_ents)
            if not overlapping and len(fn_cases) < 10:
                fn_cases.append({
                    "sentence": " ".join(words),
                    "gold_entity": " ".join(words[t_s:t_e]),
                    "gold_type": t_type,
                    "gold_span": [t_s, t_e]
                })

    error_data = {
        "false_positives": fp_cases,
        "false_negatives": fn_cases,
        "boundary_errors": boundary_cases,
        "length_f1": length_f1,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec
    }
    error_json_path = os.path.join(output_dir, "error_cases.json")
    with open(error_json_path, "w", encoding="utf-8") as f:
        json.dump(error_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {error_json_path}")
    print("=" * 70)
    print("ALL EXPERIMENTS & EVALUATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
