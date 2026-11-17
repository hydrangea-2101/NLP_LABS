"""
NLP LAB 09: Downstream Information Extraction Pipeline (NER -> Structured JSON)
-------------------------------------------------------------------------------
This production-grade module utilizes a fine-tuned Pretrained Transformer (BERT)
to parse unstructured news text, perform token-level sequence labeling, reconstruct
word/character-level entity spans, and emit structured JSON summaries.

Includes an automated evaluation test suite on 30 curated sentences with
expert gold annotations to verify extraction precision, recall, and exact match F1.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


class InformationExtractor:
    """
    End-to-end information extraction engine powered by BERT Token Classification.
    Converts raw text into structured schema with character spans and confidence scores.
    """

    def __init__(
        self,
        model_name_or_path: str = "google/bert_uncased_L-2_H-128_A-2",
        checkpoint_path: Optional[str] = "d:/My_project/NLP_LABS/lab_09/checkpoints/best_ner_model.pt",
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[InformationExtractor] Initializing tokenizer ({model_name_or_path})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        print(f"[InformationExtractor] Loading model on {self.device}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[InformationExtractor] Loading trained checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"[InformationExtractor] Warning: Checkpoint {checkpoint_path} not found. Running with base weights.")

        self.model.to(self.device)
        self.model.eval()

    def extract(self, text: str) -> Dict[str, object]:
        """
        Extracts named entities from raw text string and returns structured JSON summary.
        Tokenizes words and punctuation separately matching CoNLL-2003 standards.
        """
        import re

        # Split text into tokens (words and punctuation) preserving exact character spans
        words = []
        char_spans = []
        for match in re.finditer(r"\w+|[^\w\s]", text):
            words.append(match.group())
            char_spans.append((match.start(), match.end()))

        if not words:
            return {
                "text": text,
                "entities": [],
                "summary": {"people": [], "organizations": [], "locations": [], "misc": []}
            }

        # Tokenize with is_split_into_words=True
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        word_ids = tokenized.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [T, C]
            probs = F.softmax(logits, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()

        # Align predictions to original words using Strategy A (first subword drives prediction)
        word_labels = []
        word_confs = []
        prev_word_idx = None

        for subword_pos, w_idx in enumerate(word_ids):
            if w_idx is None:
                continue
            if w_idx != prev_word_idx:
                prev_word_idx = w_idx
                if w_idx < len(words):
                    tag = ID2LABEL[pred_indices[subword_pos]]
                    conf = float(confidences[subword_pos])
                    word_labels.append(tag)
                    word_confs.append(conf)

        # Merge B- and I- tokens into continuous entity spans
        entities = []
        curr_entity = []
        curr_type = None
        curr_start_char = None
        curr_end_char = None
        curr_confs = []

        def flush_entity():
            nonlocal curr_entity, curr_type, curr_start_char, curr_end_char, curr_confs
            if curr_entity and curr_type:
                ent_text = " ".join(curr_entity)
                avg_conf = sum(curr_confs) / len(curr_confs)
                entities.append({
                    "text": ent_text,
                    "type": curr_type,
                    "start_char": curr_start_char,
                    "end_char": curr_end_char,
                    "confidence": round(avg_conf, 4)
                })
            curr_entity = []
            curr_type = None
            curr_start_char = None
            curr_end_char = None
            curr_confs = []

        for i, (word, tag, conf) in enumerate(zip(words[:len(word_labels)], word_labels, word_confs)):
            w_start, w_end = char_spans[i]
            if tag.startswith("B-"):
                flush_entity()
                curr_type = tag[2:]
                curr_entity.append(word)
                curr_start_char = w_start
                curr_end_char = w_end
                curr_confs.append(conf)
            elif tag.startswith("I-"):
                ent_type = tag[2:]
                if curr_type == ent_type:
                    curr_entity.append(word)
                    curr_end_char = w_end
                    curr_confs.append(conf)
                else:
                    flush_entity()
                    curr_type = ent_type
                    curr_entity.append(word)
                    curr_start_char = w_start
                    curr_end_char = w_end
                    curr_confs.append(conf)
            else:
                flush_entity()

        flush_entity()

        # Categorize into structured summary
        summary = {
            "people": list(dict.fromkeys([e["text"] for e in entities if e["type"] == "PER"])),
            "organizations": list(dict.fromkeys([e["text"] for e in entities if e["type"] == "ORG"])),
            "locations": list(dict.fromkeys([e["text"] for e in entities if e["type"] == "LOC"])),
            "misc": list(dict.fromkeys([e["text"] for e in entities if e["type"] == "MISC"])),
        }

        return {
            "text": text,
            "entities": entities,
            "summary": summary
        }


# Curated benchmark of 30 news domain sentences with human gold annotations
BENCHMARK_30_SENTENCES = [
    {
        "text": "Elon Musk announced that SpaceX successfully launched 20 satellites from Florida yesterday.",
        "gold": [("Elon Musk", "PER"), ("SpaceX", "ORG"), ("Florida", "LOC")]
    },
    {
        "text": "Google CEO Sundar Pichai delivered the keynote address in Mountain View, California.",
        "gold": [("Google", "ORG"), ("Sundar Pichai", "PER"), ("Mountain View", "LOC"), ("California", "LOC")]
    },
    {
        "text": "The European Union agreed on new climate targets during the summit in Brussels.",
        "gold": [("European Union", "ORG"), ("Brussels", "LOC")]
    },
    {
        "text": "Apple is preparing to release the new iPhone across Japan and South Korea next week.",
        "gold": [("Apple", "ORG"), ("iPhone", "MISC"), ("Japan", "LOC"), ("South Korea", "LOC")]
    },
    {
        "text": "President Joe Biden met with German Chancellor Olaf Scholz at the White House in Washington.",
        "gold": [("Joe Biden", "PER"), ("German", "MISC"), ("Olaf Scholz", "PER"), ("White House", "LOC"), ("Washington", "LOC")]
    },
    {
        "text": "Microsoft agreed to invest billions into OpenAI to expand artificial intelligence research.",
        "gold": [("Microsoft", "ORG"), ("OpenAI", "ORG")]
    },
    {
        "text": "Real Madrid defeated Liverpool in the UEFA Champions League final held in Paris.",
        "gold": [("Real Madrid", "ORG"), ("Liverpool", "ORG"), ("UEFA Champions League", "MISC"), ("Paris", "LOC")]
    },
    {
        "text": "Amazon founder Jeff Bezos traveled to Tokyo to discuss supply chain partnerships.",
        "gold": [("Amazon", "ORG"), ("Jeff Bezos", "PER"), ("Tokyo", "LOC")]
    },
    {
        "text": "The World Health Organization issued a new warning from its headquarters in Geneva, Switzerland.",
        "gold": [("World Health Organization", "ORG"), ("Geneva", "LOC"), ("Switzerland", "LOC")]
    },
    {
        "text": "French President Emmanuel Macron visited London to speak with British Prime Minister Keir Starmer.",
        "gold": [("French", "MISC"), ("Emmanuel Macron", "PER"), ("London", "LOC"), ("British", "MISC"), ("Keir Starmer", "PER")]
    },
    {
        "text": "NASA launched the James Webb Space Telescope from French Guiana.",
        "gold": [("NASA", "ORG"), ("James Webb Space Telescope", "MISC"), ("French Guiana", "LOC")]
    },
    {
        "text": "Toyota announced a new electric vehicle manufacturing plant in Texas.",
        "gold": [("Toyota", "ORG"), ("Texas", "LOC")]
    },
    {
        "text": "Novak Djokovic won the Australian Open title in Melbourne after a five-set battle.",
        "gold": [("Novak Djokovic", "PER"), ("Australian Open", "MISC"), ("Melbourne", "LOC")]
    },
    {
        "text": "The Bank of England raised interest rates following inflation data in the United Kingdom.",
        "gold": [("Bank of England", "ORG"), ("United Kingdom", "LOC")]
    },
    {
        "text": "Jensen Huang presented the latest Blackwell architecture at the Nvidia GTC conference in San Jose.",
        "gold": [("Jensen Huang", "PER"), ("Blackwell", "MISC"), ("Nvidia", "ORG"), ("GTC", "MISC"), ("San Jose", "LOC")]
    },
    {
        "text": "The United Nations General Assembly convened in New York to address global security.",
        "gold": [("United Nations General Assembly", "ORG"), ("New York", "LOC")]
    },
    {
        "text": "Samsung Electronics posted record profits driven by semiconductor demand in Asia.",
        "gold": [("Samsung Electronics", "ORG"), ("Asia", "LOC")]
    },
    {
        "text": "Barack Obama and Michelle Obama attended the film festival in Chicago.",
        "gold": [("Barack Obama", "PER"), ("Michelle Obama", "PER"), ("Chicago", "LOC")]
    },
    {
        "text": "Ferrari revealed its latest Formula One car at an exhibition in Milan, Italy.",
        "gold": [("Ferrari", "ORG"), ("Formula One", "MISC"), ("Milan", "LOC"), ("Italy", "LOC")]
    },
    {
        "text": "The Federal Reserve chairman spoke before the Senate Banking Committee in Washington.",
        "gold": [("Federal Reserve", "ORG"), ("Senate Banking Committee", "ORG"), ("Washington", "LOC")]
    },
    {
        "text": "Sony announced PlayStation sales reached new milestones across North America.",
        "gold": [("Sony", "ORG"), ("PlayStation", "MISC"), ("North America", "LOC")]
    },
    {
        "text": "Cristiano Ronaldo scored twice for Al Nassr in Riyadh, Saudi Arabia.",
        "gold": [("Cristiano Ronaldo", "PER"), ("Al Nassr", "ORG"), ("Riyadh", "LOC"), ("Saudi Arabia", "LOC")]
    },
    {
        "text": "The International Monetary Fund published its economic outlook in Washington DC.",
        "gold": [("International Monetary Fund", "ORG"), ("Washington DC", "LOC")]
    },
    {
        "text": "Singapore Airlines expanded its flight network to Sydney and Auckland.",
        "gold": [("Singapore Airlines", "ORG"), ("Sydney", "LOC"), ("Auckland", "LOC")]
    },
    {
        "text": "Taylor Swift concluded her world tour with three sold-out concerts in Toronto, Canada.",
        "gold": [("Taylor Swift", "PER"), ("Toronto", "LOC"), ("Canada", "LOC")]
    },
    {
        "text": "Intel unveiled plans to construct a massive semiconductor fabrication facility in Ohio.",
        "gold": [("Intel", "ORG"), ("Ohio", "LOC")]
    },
    {
        "text": "The British Museum in London returned several historical artifacts to Greece.",
        "gold": [("British Museum", "ORG"), ("London", "LOC"), ("Greece", "LOC")]
    },
    {
        "text": "Bill Gates delivered an inspiring lecture on global philanthropy at Harvard University in Cambridge.",
        "gold": [("Bill Gates", "PER"), ("Harvard University", "ORG"), ("Cambridge", "LOC")]
    },
    {
        "text": "The Olympic Games will open in Los Angeles following extensive stadium renovations.",
        "gold": [("Olympic Games", "MISC"), ("Los Angeles", "LOC")]
    },
    {
        "text": "Pfizer and BioNTech collaborated to produce innovative mRNA treatments in Germany.",
        "gold": [("Pfizer", "ORG"), ("BioNTech", "ORG"), ("Germany", "LOC")]
    }
]


def evaluate_on_benchmark(extractor: InformationExtractor) -> Dict[str, float]:
    """
    Evaluates the extractor on the curated 30-sentence benchmark against human gold standards.
    Measures exact match True Positives, False Positives, and False Negatives.
    """
    total_gold = 0
    total_pred = 0
    tp = 0
    fp = 0
    fn = 0

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION (30 News Domain Sentences)")
    print("=" * 70)

    for i, item in enumerate(BENCHMARK_30_SENTENCES, 1):
        res = extractor.extract(item["text"])
        pred_tuples = set((e["text"].lower(), e["type"]) for e in res["entities"])
        gold_tuples = set((text.lower(), ent_type) for text, ent_type in item["gold"])

        total_gold += len(gold_tuples)
        total_pred += len(pred_tuples)

        sentence_tp = len(pred_tuples.intersection(gold_tuples))
        sentence_fp = len(pred_tuples - gold_tuples)
        sentence_fn = len(gold_tuples - pred_tuples)

        tp += sentence_tp
        fp += sentence_fp
        fn += sentence_fn

        if i <= 5 or sentence_fp > 0 or sentence_fn > 0:
            print(f"Sentence [{i:02d}]: {item['text'][:65]}...")
            print(f"  Gold: {list(gold_tuples)}")
            print(f"  Pred: {list(pred_tuples)}")
            if sentence_fp > 0 or sentence_fn > 0:
                print(f"  Diff -> FP: {list(pred_tuples - gold_tuples)} | FN: {list(gold_tuples - pred_tuples)}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("-" * 70)
    print(f"Total Gold Entities:      {total_gold}")
    print(f"Total Predicted Entities: {total_pred}")
    print(f"True Positives (TP):      {tp}")
    print(f"False Positives (FP):     {fp}")
    print(f"False Negatives (FN):     {fn}")
    print(f"Benchmark Precision:      {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Benchmark Recall:         {recall:.4f} ({recall * 100:.2f}%)")
    print(f"Benchmark F1-Score:       {f1:.4f} ({f1 * 100:.2f}%)")
    print("=" * 70)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def main():
    parser = argparse.ArgumentParser(description="Information Extraction Pipeline")
    parser.add_argument("--checkpoint", type=str, default="d:/My_project/NLP_LABS/lab_09/checkpoints/best_ner_model.pt")
    parser.add_argument("--eval_benchmark", action="store_true", default=True)
    args = parser.parse_args()

    extractor = InformationExtractor(checkpoint_path=args.checkpoint)

    # Interactive sample demonstration
    sample_text = "Elon Musk announced that SpaceX successfully launched 20 satellites from Florida yesterday."
    result = extractor.extract(sample_text)
    print("\nSAMPLE EXTRACTION OUTPUT (Structured JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.eval_benchmark:
        evaluate_on_benchmark(extractor)


if __name__ == "__main__":
    main()
