"""
NLP LAB 08 — PRETRAINED TRANSFORMERS AND BERT FINE-TUNING
Module cốt lõi triển khai toàn diện quy trình Fine-tuning, Thử nghiệm so sánh,
Đánh giá định lượng, Phân tích độ tin cậy và Phân tích lỗi.

Mô hình lựa chọn: google/bert_uncased_L-2_H-128_A-2 (BERT-Tiny, 4.39M tham số)
Lý do lựa chọn:
- Môi trường thực thi CPU (16 logical cores, không có GPU CUDA rời).
- Giữ trọn vẹn kiến trúc chuẩn của BERT (WordPiece 30k vocab, bidirectional encoder, CLS classification head).
- Thời gian huấn luyện nhanh (~15s/epoch trên CPU), an toàn bộ nhớ RAM (< 500MB), tính tái lặp 100%.
"""

import os
import sys
import time
import math
import json
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any

# Đảm bảo mã hóa UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification
)

# ==============================================================================
# 1. THIẾT LẬP REPRODUCIBILITY & HARDWARE
# ==============================================================================
SEED = 42

def set_seed(seed: int = SEED):
    """Cố định tất cả random seeds để đảm bảo tính tái lập 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_current_memory_mb() -> float:
    """Đo lượng bộ nhớ RAM (RSS) tiến trình hiện tại đang sử dụng (MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# ==============================================================================
# 2. XỬ LÝ VÀ CHUẨN BỊ DỮ LIỆU
# ==============================================================================
class TextClassificationDataset(Dataset):
    """
    Dataset tương thích với PyTorch DataLoader, tích hợp sẵn Hugging Face Tokenizer.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def load_and_split_imdb(data_path: str, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nạp dữ liệu IMDb, loại bỏ trùng lặp, chia tập train/val/test phân tầng
    và trích xuất tập mẫu chuẩn (5,000 train, 1,000 val, 1,000 test)
    hoàn toàn nhất quán với LAB 05, LAB 06 và LAB 07.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu IMDb tại: {data_path}")
        
    df_raw = pd.read_csv(data_path)
    df = df_raw.drop_duplicates(subset=['review']).reset_index(drop=True)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # 80% train, 20% temp (10% val, 10% test)
    train_full, temp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df['label']
    )
    val_full, test_full = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df['label']
    )
    
    # Tập mẫu chuẩn cân bằng nhãn: 5,000 Train, 1,000 Val, 1,000 Test
    train_subset = pd.concat([
        train_full[train_full['label'] == 1].iloc[:2500],
        train_full[train_full['label'] == 0].iloc[:2500]
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    val_subset = pd.concat([
        val_full[val_full['label'] == 1].iloc[:500],
        val_full[val_full['label'] == 0].iloc[:500]
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    test_subset = pd.concat([
        test_full[test_full['label'] == 1].iloc[:500],
        test_full[test_full['label'] == 0].iloc[:500]
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    return train_subset, val_subset, test_subset

def load_amazon_domain_data(data_path: str) -> pd.DataFrame:
    """Nạp tập dữ liệu ngoài miền Amazon Product Reviews từ UCI."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu Amazon tại: {data_path}")
    df_amazon = pd.read_csv(data_path, sep='\t', header=None, names=['text', 'label'])
    df_amazon['label'] = df_amazon['label'].astype(int)
    return df_amazon

# ==============================================================================
# 3. XÂY DỰNG VÀ QUẢN LÝ MÔ HÌNH TRANSFORMER
# ==============================================================================
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"

def build_model(
    model_name: str = MODEL_NAME,
    mode: str = "fine_tuning",
    from_scratch: bool = False,
    num_labels: int = 2
) -> nn.Module:
    """
    Khởi tạo mô hình Transformer cho phân loại văn bản:
    - mode="fine_tuning": Toàn bộ encoder và classification head đều được cập nhật gradient.
    - mode="frozen": Đóng băng pretrained encoder, chỉ cập nhật classification head.
    - from_scratch=True: Giữ nguyên kiến trúc nhưng khởi tạo trọng số ngẫu nhiên (không dùng pretraining).
    """
    if from_scratch:
        # Khởi tạo mô hình kiến trúc BERT từ cấu hình ngẫu nhiên (from scratch)
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        model = BertForSequenceClassification(config)
    else:
        # Nạp trọng số Pretrained từ Hugging Face Hub
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
    if mode == "frozen":
        # Đóng băng toàn bộ encoder backbone
        for param in model.bert.parameters():
            param.requires_grad = False
            
    return model

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Trả về (tổng số tham số, số tham số có thể huấn luyện trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ==============================================================================
# 4. HUẤN LUYỆN, ĐÁNH GIÁ VÀ LƯU CHECKPOINT
# ==============================================================================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Huấn luyện 1 epoch, trả về (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Đánh giá toàn diện trên tập dữ liệu, trả về loss, metrics, xác suất và nhãn."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item() * len(labels)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs)
    }

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str):
    """Lưu checkpoint mô hình an toàn."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None
    }, filepath)

def load_checkpoint(model: nn.Module, filepath: str):
    """Nạp trọng số checkpoint đã lưu."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

# ==============================================================================
# 5. CÁC THỰC NGHIỆM CHÍNH (EXPERIMENTS 1 - 5)
# ==============================================================================
def run_training_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 3,
    lr: float = 5e-5,
    checkpoint_path: Optional[str] = None
) -> Tuple[Dict[str, Any], float]:
    """Chạy quy trình huấn luyện và đánh giá một thiết lập thử nghiệm."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_val_f1 = -1.0
    
    start_time = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        val_eval = evaluate(model, val_loader, device)
        if val_eval["f1"] > best_val_f1 and checkpoint_path:
            best_val_f1 = val_eval["f1"]
            save_checkpoint(model, optimizer, ep, checkpoint_path)
            
    train_time = time.time() - start_time
    
    # Nạp best model để đánh giá trên test nếu có checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(model, checkpoint_path)
        
    test_eval = evaluate(model, test_loader, device)
    return test_eval, train_time

# ==============================================================================
# 6. PIPELINE TỔNG HỢP TOÀN BỘ LAB 08
# ==============================================================================
def run_all_lab08_experiments(
    imdb_path: str = "D:/My_project/NLP_LABS/data/IMDB Dataset.csv",
    amazon_path: str = "D:/My_project/NLP_LABS/data/amazon_cells_labelled.txt",
    output_dir: str = "D:/My_project/NLP_LABS/lab_08"
):
    """
    Thực hiện trọn vẹn 5 thực nghiệm của LAB 08:
    1. Pretrained vs From-Scratch
    2. Feature Extraction (Frozen) vs Fine-Tuning
    3. Data Efficiency (10%, 25%, 50%, 100%)
    4. Maximum Sequence Length (64, 128, 256)
    5. Domain Shift (IMDb vs Amazon Product Reviews)
    Và xuất kết quả vào results.csv, biểu đồ PNG, error_analysis.md.
    """
    print("=" * 70)
    print("   BẮT ĐẦU THỰC HIỆN TOÀN BỘ CÁC THỰC NGHIỆM ĐỊNH LƯỢNG LAB 08")
    print(f"   Thiết bị: {device} | Bộ nhớ RAM ban đầu: {get_current_memory_mb():.1f} MB")
    print("=" * 70)
    
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. Nạp và phân chia dữ liệu
    print("\n[*] Nạp tập dữ liệu IMDb...")
    train_df, val_df, test_df = load_and_split_imdb(imdb_path, seed=SEED)
    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Chuẩn bị DataLoaders mặc định (max_length=128, batch_size=32)
    val_ds = TextClassificationDataset(val_df['review'], val_df['label'], tokenizer, max_length=128)
    test_ds = TextClassificationDataset(test_df['review'], test_df['label'], tokenizer, max_length=128)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    train_ds_100 = TextClassificationDataset(train_df['review'], train_df['label'], tokenizer, max_length=128)
    train_loader_100 = DataLoader(train_ds_100, batch_size=32, shuffle=True)
    
    results_records = []
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # THỰC NGHIỆM 1: Pretrained vs From Scratch
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("[EXPERIMENT 1] Pretrained vs From Scratch")
    print("=" * 50)
    
    # Model A: From Scratch
    print("[1/2] Huấn luyện Model A: Transformer From Scratch (Random Weights)...")
    model_scratch = build_model(from_scratch=True, mode="fine_tuning")
    tot_params, train_params = count_parameters(model_scratch)
    ckpt_scratch = os.path.join(checkpoints_dir, "scratch_model.pt")
    eval_scratch, time_scratch = run_training_experiment(
        model_scratch, train_loader_100, val_loader, test_loader,
        epochs=3, lr=1e-4, checkpoint_path=ckpt_scratch
    )
    print(f"      -> Scratch: Acc={eval_scratch['accuracy']:.4f}, F1={eval_scratch['f1']:.4f}, Time={time_scratch:.2f}s")
    results_records.append({
        "Experiment": "Exp1_Pretrained_vs_Scratch",
        "Model": "Transformer From Scratch",
        "Condition": "Random Init (100% Data)",
        "Trainable Params": train_params,
        "Total Params": tot_params,
        "Accuracy": eval_scratch["accuracy"],
        "Precision": eval_scratch["precision"],
        "Recall": eval_scratch["recall"],
        "F1": eval_scratch["f1"],
        "Training Time (s)": round(time_scratch, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # Model B: Pretrained BERT Fine-tuning
    print("[2/2] Huấn luyện Model B: Pretrained BERT Fine-tuning...")
    model_pretrained = build_model(from_scratch=False, mode="fine_tuning")
    tot_params_pt, train_params_pt = count_parameters(model_pretrained)
    ckpt_pretrained = os.path.join(checkpoints_dir, "pretrained_finetuned.pt")
    eval_pretrained, time_pretrained = run_training_experiment(
        model_pretrained, train_loader_100, val_loader, test_loader,
        epochs=3, lr=5e-5, checkpoint_path=ckpt_pretrained
    )
    print(f"      -> Pretrained: Acc={eval_pretrained['accuracy']:.4f}, F1={eval_pretrained['f1']:.4f}, Time={time_pretrained:.2f}s")
    results_records.append({
        "Experiment": "Exp1_Pretrained_vs_Scratch",
        "Model": "Pretrained BERT Fine-Tuned",
        "Condition": "Pretrained Weights (100% Data)",
        "Trainable Params": train_params_pt,
        "Total Params": tot_params_pt,
        "Accuracy": eval_pretrained["accuracy"],
        "Precision": eval_pretrained["precision"],
        "Recall": eval_pretrained["recall"],
        "F1": eval_pretrained["f1"],
        "Training Time (s)": round(time_pretrained, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # --------------------------------------------------------------------------
    # THỰC NGHIỆM 2: Feature Extraction (Frozen) vs Fine-Tuning
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("[EXPERIMENT 2] Feature Extraction (Frozen) vs Fine-Tuning")
    print("=" * 50)
    
    print("[1/2] Huấn luyện Mode A: Frozen Encoder (Feature Extraction)...")
    model_frozen = build_model(from_scratch=False, mode="frozen")
    tot_params_frz, train_params_frz = count_parameters(model_frozen)
    assert train_params_frz == 258, f"Kỳ vọng 258 tham số phân loại nhưng nhận {train_params_frz}"
    ckpt_frozen = os.path.join(checkpoints_dir, "frozen_model.pt")
    eval_frozen, time_frozen = run_training_experiment(
        model_frozen, train_loader_100, val_loader, test_loader,
        epochs=3, lr=5e-4, checkpoint_path=ckpt_frozen
    )
    print(f"      -> Frozen: Acc={eval_frozen['accuracy']:.4f}, F1={eval_frozen['f1']:.4f}, Time={time_frozen:.2f}s")
    results_records.append({
        "Experiment": "Exp2_Frozen_vs_FineTuning",
        "Model": "Pretrained BERT (Frozen Encoder)",
        "Condition": "Linear Head Only Trainable (258 params)",
        "Trainable Params": train_params_frz,
        "Total Params": tot_params_frz,
        "Accuracy": eval_frozen["accuracy"],
        "Precision": eval_frozen["precision"],
        "Recall": eval_frozen["recall"],
        "F1": eval_frozen["f1"],
        "Training Time (s)": round(time_frozen, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # Mode B đã có từ Exp 1, ghi lại bản ghi cho Exp 2
    results_records.append({
        "Experiment": "Exp2_Frozen_vs_FineTuning",
        "Model": "Pretrained BERT (Full Fine-Tuning)",
        "Condition": "All Layers Trainable (4.39M params)",
        "Trainable Params": train_params_pt,
        "Total Params": tot_params_pt,
        "Accuracy": eval_pretrained["accuracy"],
        "Precision": eval_pretrained["precision"],
        "Recall": eval_pretrained["recall"],
        "F1": eval_pretrained["f1"],
        "Training Time (s)": round(time_pretrained, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # --------------------------------------------------------------------------
    # THỰC NGHIỆM 3: Amount of Training Data (Data Efficiency)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("[EXPERIMENT 3] Data Efficiency (10%, 25%, 50%, 100%)")
    print("=" * 50)
    
    data_efficiency_results = {
        "100%": eval_pretrained
    }
    
    subset_fractions = [0.10, 0.25, 0.50]
    for frac in subset_fractions:
        n_samples = int(len(train_df) * frac)
        sub_df = pd.concat([
            train_df[train_df['label'] == 1].iloc[:n_samples // 2],
            train_df[train_df['label'] == 0].iloc[:n_samples // 2]
        ]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        
        print(f"[*] Huấn luyện với {int(frac*100)}% dữ liệu ({len(sub_df)} mẫu)...")
        sub_ds = TextClassificationDataset(sub_df['review'], sub_df['label'], tokenizer, max_length=128)
        sub_loader = DataLoader(sub_ds, batch_size=32, shuffle=True)
        
        m = build_model(from_scratch=False, mode="fine_tuning")
        eval_sub, time_sub = run_training_experiment(
            m, sub_loader, val_loader, test_loader, epochs=3, lr=5e-5
        )
        print(f"    -> {int(frac*100)}%: Acc={eval_sub['accuracy']:.4f}, F1={eval_sub['f1']:.4f}, Time={time_sub:.2f}s")
        data_efficiency_results[f"{int(frac*100)}%"] = eval_sub
        
        results_records.append({
            "Experiment": "Exp3_Data_Efficiency",
            "Model": f"Pretrained BERT ({int(frac*100)}% Data)",
            "Condition": f"{len(sub_df)} Train Samples",
            "Trainable Params": train_params_pt,
            "Total Params": tot_params_pt,
            "Accuracy": eval_sub["accuracy"],
            "Precision": eval_sub["precision"],
            "Recall": eval_sub["recall"],
            "F1": eval_sub["f1"],
            "Training Time (s)": round(time_sub, 2),
            "Memory (MB)": round(get_current_memory_mb(), 1)
        })
        
    results_records.append({
        "Experiment": "Exp3_Data_Efficiency",
        "Model": "Pretrained BERT (100% Data)",
        "Condition": f"{len(train_df)} Train Samples",
        "Trainable Params": train_params_pt,
        "Total Params": tot_params_pt,
        "Accuracy": eval_pretrained["accuracy"],
        "Precision": eval_pretrained["precision"],
        "Recall": eval_pretrained["recall"],
        "F1": eval_pretrained["f1"],
        "Training Time (s)": round(time_pretrained, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # --------------------------------------------------------------------------
    # THỰC NGHIỆM 4: Maximum Sequence Length (64, 128, 256)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("[EXPERIMENT 4] Maximum Sequence Length (64, 128, 256)")
    print("=" * 50)
    
    seq_lengths = [64, 256]
    for seq_len in seq_lengths:
        print(f"[*] Huấn luyện với max_length = {seq_len}...")
        tr_ds_seq = TextClassificationDataset(train_df['review'], train_df['label'], tokenizer, max_length=seq_len)
        va_ds_seq = TextClassificationDataset(val_df['review'], val_df['label'], tokenizer, max_length=seq_len)
        te_ds_seq = TextClassificationDataset(test_df['review'], test_df['label'], tokenizer, max_length=seq_len)
        
        tr_ld_seq = DataLoader(tr_ds_seq, batch_size=32, shuffle=True)
        va_ld_seq = DataLoader(va_ds_seq, batch_size=32, shuffle=False)
        te_ld_seq = DataLoader(te_ds_seq, batch_size=32, shuffle=False)
        
        m_seq = build_model(from_scratch=False, mode="fine_tuning")
        eval_seq, time_seq = run_training_experiment(
            m_seq, tr_ld_seq, va_ld_seq, te_ld_seq, epochs=3, lr=5e-5
        )
        mem_seq = get_current_memory_mb()
        print(f"    -> T={seq_len}: Acc={eval_seq['accuracy']:.4f}, F1={eval_seq['f1']:.4f}, Time={time_seq:.2f}s, Mem={mem_seq:.1f}MB")
        
        results_records.append({
            "Experiment": "Exp4_Sequence_Length",
            "Model": f"Pretrained BERT (T={seq_len})",
            "Condition": f"max_length={seq_len}",
            "Trainable Params": train_params_pt,
            "Total Params": tot_params_pt,
            "Accuracy": eval_seq["accuracy"],
            "Precision": eval_seq["precision"],
            "Recall": eval_seq["recall"],
            "F1": eval_seq["f1"],
            "Training Time (s)": round(time_seq, 2),
            "Memory (MB)": round(mem_seq, 1)
        })
        
    results_records.append({
        "Experiment": "Exp4_Sequence_Length",
        "Model": "Pretrained BERT (T=128)",
        "Condition": "max_length=128",
        "Trainable Params": train_params_pt,
        "Total Params": tot_params_pt,
        "Accuracy": eval_pretrained["accuracy"],
        "Precision": eval_pretrained["precision"],
        "Recall": eval_pretrained["recall"],
        "F1": eval_pretrained["f1"],
        "Training Time (s)": round(time_pretrained, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # --------------------------------------------------------------------------
    # THỰC NGHIỆM 5: Domain Shift (Movie Reviews vs Amazon Product Reviews)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("[EXPERIMENT 5] Domain Shift (In-domain IMDb vs Out-of-domain Amazon)")
    print("=" * 50)
    
    df_amazon = load_amazon_domain_data(amazon_path)
    print(f"[*] Nạp thành công {len(df_amazon)} mẫu ngoài miền từ Amazon Product Reviews.")
    amazon_ds = TextClassificationDataset(df_amazon['text'], df_amazon['label'], tokenizer, max_length=128)
    amazon_loader = DataLoader(amazon_ds, batch_size=32, shuffle=False)
    
    # Đánh giá best fine-tuned model trên tập ngoài miền Amazon
    eval_amazon = evaluate(model_pretrained, amazon_loader, device)
    print(f"    -> In-Domain (IMDb Movie Reviews):     Acc={eval_pretrained['accuracy']:.4f}, F1={eval_pretrained['f1']:.4f}")
    print(f"    -> Out-Of-Domain (Amazon Product Reviews): Acc={eval_amazon['accuracy']:.4f}, F1={eval_amazon['f1']:.4f}")
    
    results_records.append({
        "Experiment": "Exp5_Domain_Shift",
        "Model": "Pretrained BERT Fine-Tuned",
        "Condition": "In-Domain (IMDb Movie Reviews)",
        "Trainable Params": train_params_pt,
        "Total Params": tot_params_pt,
        "Accuracy": eval_pretrained["accuracy"],
        "Precision": eval_pretrained["precision"],
        "Recall": eval_pretrained["recall"],
        "F1": eval_pretrained["f1"],
        "Training Time (s)": round(time_pretrained, 2),
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    results_records.append({
        "Experiment": "Exp5_Domain_Shift",
        "Model": "Pretrained BERT Fine-Tuned",
        "Condition": "Out-of-Domain (Amazon Product Reviews)",
        "Trainable Params": train_params_pt,
        "Total Params": tot_params_pt,
        "Accuracy": eval_amazon["accuracy"],
        "Precision": eval_amazon["precision"],
        "Recall": eval_amazon["recall"],
        "F1": eval_amazon["f1"],
        "Training Time (s)": 0.0,
        "Memory (MB)": round(get_current_memory_mb(), 1)
    })
    
    # --------------------------------------------------------------------------
    # 7. XUẤT KẾT QUẢ RESULTS.CSV
    # --------------------------------------------------------------------------
    res_df = pd.DataFrame(results_records)
    results_csv_path = os.path.join(output_dir, "results.csv")
    res_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    print(f"\n[+] Đã lưu toàn bộ số liệu vào {results_csv_path}")
    
    # --------------------------------------------------------------------------
    # 8. VẼ VÀ LƯU CÁC BIỂU ĐỒ TRỰC QUAN HÓA
    # --------------------------------------------------------------------------
    print("\n[*] Đang tạo các biểu đồ trực quan hóa...")
    
    # A. Confusion Matrix (confusion_matrix.png)
    cm = eval_pretrained["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix — Pretrained BERT Fine-Tuning (Test Set)", fontsize=11, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negative (0)", "Positive (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Negative (0)", "Positive (1)"], fontsize=10)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / cm.sum() * 100
            plt.text(j, i, f"{val}\n({pct:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=11, fontweight='bold')
                     
    plt.ylabel("True Label", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu {cm_path}")
    
    # B. Data Efficiency Curve (data_efficiency.png)
    percentages = [10, 25, 50, 100]
    accs = [
        data_efficiency_results["10%"]["accuracy"],
        data_efficiency_results["25%"]["accuracy"],
        data_efficiency_results["50%"]["accuracy"],
        data_efficiency_results["100%"]["accuracy"]
    ]
    f1s = [
        data_efficiency_results["10%"]["f1"],
        data_efficiency_results["25%"]["f1"],
        data_efficiency_results["50%"]["f1"],
        data_efficiency_results["100%"]["f1"]
    ]
    
    plt.figure(figsize=(7, 5))
    plt.plot(percentages, accs, marker='o', linewidth=2.2, color='#1f77b4', label='Accuracy')
    plt.plot(percentages, f1s, marker='s', linewidth=2.2, linestyle='--', color='#2ca02c', label='F1-Score')
    for x, y in zip(percentages, accs):
        plt.text(x, y + 0.008, f"{y:.3f}", ha='center', fontsize=9, fontweight='bold', color='#1f77b4')
    for x, y in zip(percentages, f1s):
        plt.text(x, y - 0.015, f"{y:.3f}", ha='center', fontsize=9, fontweight='bold', color='#2ca02c')
        
    plt.title("Learning & Data-Efficiency Curve (Pretrained BERT)", fontsize=11, fontweight='bold', pad=12)
    plt.xlabel("Tỷ lệ dữ liệu huấn luyện (%)", fontsize=10, fontweight='bold')
    plt.ylabel("Điểm số Đánh giá (Score)", fontsize=10, fontweight='bold')
    plt.xticks(percentages, ["10% (500)", "25% (1250)", "50% (2500)", "100% (5000)"], fontsize=9)
    plt.ylim(min(accs + f1s) - 0.05, max(accs + f1s) + 0.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    de_path = os.path.join(output_dir, "data_efficiency.png")
    plt.savefig(de_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu {de_path}")
    
    # C. Confidence Analysis (confidence_analysis.png)
    test_probs = eval_pretrained["probs"]
    test_preds = eval_pretrained["preds"]
    test_labels = eval_pretrained["labels"]
    confidences = np.max(test_probs, axis=1)
    is_correct = (test_preds == test_labels)
    
    # Phân nhóm độ tin cậy: Low [0.5, 0.7), Medium [0.7, 0.9), High [0.9, 1.0]
    bins = [0.50, 0.70, 0.90, 1.00]
    bin_names = ["Low [0.5, 0.7)", "Medium [0.7, 0.9)", "High [0.9, 1.0]"]
    bin_counts = []
    bin_accs = []
    
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (confidences >= bins[i]) & (confidences <= bins[i+1])
        else:
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            
        count = int(np.sum(mask))
        acc = float(np.mean(is_correct[mask])) if count > 0 else 0.0
        bin_counts.append(count)
        bin_accs.append(acc)
        
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(bin_names))
    width = 0.4
    
    color_bar = '#3498db'
    color_line = '#e74c3c'
    
    bars = ax1.bar(x_pos - width/2, bin_counts, width, label='Số lượng mẫu (Count)', color=color_bar, alpha=0.85)
    ax1.set_xlabel('Khoảng độ tin cậy (Confidence Bin)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Số lượng mẫu', color=color_bar, fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bin_names, fontsize=9)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
                     
    ax2 = ax1.twinx()
    lines = ax2.plot(x_pos + width/2, bin_accs, color=color_line, marker='o', linewidth=2.5, label='Độ chính xác thực tế (Accuracy)')
    ax2.set_ylabel('Độ chính xác thực tế', color=color_line, fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim(0.4, 1.05)
    
    for x, y in zip(x_pos + width/2, bin_accs):
        ax2.annotate(f'{y*100:.1f}%',
                     xy=(x, y), xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color=color_line)
                     
    plt.title("Phân tích Độ tin cậy (Confidence Analysis) vs Độ chính xác thực tế", fontsize=11, fontweight='bold', pad=12)
    fig.tight_layout()
    conf_path = os.path.join(output_dir, "confidence_analysis.png")
    plt.savefig(conf_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu {conf_path}")
    
    # --------------------------------------------------------------------------
    # 9. TRÍCH XUẤT CÁC CA LỖI (ERROR ANALYSIS EXAMPLES)
    # --------------------------------------------------------------------------
    print("\n[*] Đang tự động trích xuất các ca lỗi thực tế cho error_analysis.md...")
    
    # 3 False Positives: true=0 (Negative), pred=1 (Positive)
    fp_indices = np.where((test_labels == 0) & (test_preds == 1))[0]
    # Sắp xếp theo confidence giảm dần
    fp_sorted = fp_indices[np.argsort(-confidences[fp_indices])]
    
    # 3 False Negatives: true=1 (Positive), pred=0 (Negative)
    fn_indices = np.where((test_labels == 1) & (test_preds == 0))[0]
    fn_sorted = fn_indices[np.argsort(-confidences[fn_indices])]
    
    # 2 High-confidence errors: toàn bộ ca sai có confidence cao nhất
    all_err_indices = np.where(test_preds != test_labels)[0]
    all_err_sorted = all_err_indices[np.argsort(-confidences[all_err_indices])]
    
    extracted_errors = {
        "false_positives": [],
        "false_negatives": [],
        "high_confidence_errors": []
    }
    
    for idx in fp_sorted[:3]:
        extracted_errors["false_positives"].append({
            "index": int(idx),
            "text": str(test_df["review"].iloc[idx]),
            "true_label": 0,
            "pred_label": 1,
            "confidence": float(confidences[idx])
        })
        
    for idx in fn_sorted[:3]:
        extracted_errors["false_negatives"].append({
            "index": int(idx),
            "text": str(test_df["review"].iloc[idx]),
            "true_label": 1,
            "pred_label": 0,
            "confidence": float(confidences[idx])
        })
        
    for idx in all_err_sorted[:2]:
        extracted_errors["high_confidence_errors"].append({
            "index": int(idx),
            "text": str(test_df["review"].iloc[idx]),
            "true_label": int(test_labels[idx]),
            "pred_label": int(test_preds[idx]),
            "confidence": float(confidences[idx])
        })
        
    err_json_path = os.path.join(output_dir, "error_cases.json")
    with open(err_json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_errors, f, ensure_ascii=False, indent=2)
    print(f"    -> Đã lưu các mẫu lỗi vào {err_json_path}")
    
    print("\n" + "=" * 70)
    print("   HOÀN THÀNH TOÀN BỘ 5 THỰC NGHIỆM ĐỊNH LƯỢNG LAB 08 THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP LAB 08 Pretrained Transformers Runner")
    parser.add_argument("--imdb_path", type=str, default="D:/My_project/NLP_LABS/data/IMDB Dataset.csv")
    parser.add_argument("--amazon_path", type=str, default="D:/My_project/NLP_LABS/data/amazon_cells_labelled.txt")
    parser.add_argument("--output_dir", type=str, default="D:/My_project/NLP_LABS/lab_08")
    args = parser.parse_args()
    
    run_all_lab08_experiments(
        imdb_path=args.imdb_path,
        amazon_path=args.amazon_path,
        output_dir=args.output_dir
    )
