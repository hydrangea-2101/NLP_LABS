"""
NLP LAB 07 — TRANSFORMER ARCHITECTURE
Script thực hiện toàn bộ các thực nghiệm định lượng của LAB 07:
1. Experiment 1 & Permutation Experiment (Mục 11, 12) -> lưu positional_encoding.png
2. Experiment 2: Number of attention heads n_heads in {1, 2, 4, 8} (Mục 19)
3. Experiment 3: Attention head diversity visualization (Mục 20) -> lưu attention_heads.png
4. Experiment 4: Transformer vs RNN/LSTM trên cùng tập dữ liệu IMDb (Mục 21)
5. Experiment 5: Sequence length max_length in {32, 64, 128, 256} & chi phí O(T^2) (Mục 22)
6. Application: Phân loại văn bản trên 5 nhóm câu kiểm thử (Mục 23)
7. Experiment 6: Pooling strategy — Mean pooling vs CLS-style first-token pooling (Mục 25)
8. Learning curves (Train/Val Loss & Accuracy theo epoch — Mục 29) -> lưu learning_curves.png
9. Confusion Matrix trên tập Test -> lưu confusion_matrix.png
10. Trích xuất 3 ca đúng & 3 ca sai kèm phân tích Attention weights cho error_analysis.md
11. Xuất bảng tổng hợp kết quả định lượng vào results.csv và lab07_summary.json
"""

import os
import re
import sys
import math
import time
import json
import random
from collections import Counter

# Đảm bảo mã hóa UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

from positional_encoding import (
    PositionalEncoding,
    run_section_10_calculation,
    save_positional_encoding_plot
)
from transformer_block import (
    TransformerBlock,
    verify_shape_section_18,
    verify_hand_calculations
)

# ==============================================================================
# 1. THIẾT LẬP REPRODUCIBILITY & THIẾT BỊ
# ==============================================================================
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Thiết bị tính toán: {device}")

# Chạy kiểm tra Mục 10, 11, 12, 18 trước
run_section_10_calculation()
save_positional_encoding_plot("positional_encoding.png")
verify_shape_section_18()
verify_hand_calculations()

# ==============================================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU IMDB (NHẤT QUÁN VỚI LAB 05 & LAB 06)
# ==============================================================================
DATA_PATH = os.path.join("..", "data", "IMDB Dataset.csv")
assert os.path.exists(DATA_PATH), f"Không tìm thấy dữ liệu tại {DATA_PATH}"

df_raw = pd.read_csv(DATA_PATH)
df = df_raw.drop_duplicates(subset=['review']).reset_index(drop=True)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label'])

# Cùng tập mẫu chuẩn với LAB 05 & LAB 06: 5000 Train, 1000 Val, 1000 Test
train_subset = pd.concat([
    train_df[train_df['label'] == 1].iloc[:2500],
    train_df[train_df['label'] == 0].iloc[:2500]
]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

val_subset = pd.concat([
    val_df[val_df['label'] == 1].iloc[:500],
    val_df[val_df['label'] == 0].iloc[:500]
]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

test_subset = pd.concat([
    test_df[test_df['label'] == 1].iloc[:500],
    test_df[test_df['label'] == 0].iloc[:500]
]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(f"[*] Kích thước tập dữ liệu — Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

class Vocabulary:
    def __init__(self, max_size: int = 15000):
        self.max_size = max_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<CLS>'}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        for word, _ in counter.most_common(self.max_size - len(self.word2idx)):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 128, prepend_cls: bool = False):
        tokens = text.split()
        raw_len = len(tokens)
        if prepend_cls:
            usable_tokens = tokens[:max_len - 1]
            ids = [self.word2idx['<CLS>']] + [self.word2idx.get(w, 1) for w in usable_tokens]
        else:
            usable_tokens = tokens[:max_len]
            ids = [self.word2idx.get(w, 1) for w in usable_tokens]
        retained_len = len(ids)
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids, retained_len, raw_len

vocab = Vocabulary(max_size=15000)
vocab.build(train_subset['clean_review'].tolist())
print(f"[*] Kích thước từ điển (Vocab size): {len(vocab.word2idx)}")


class IMDbDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Vocabulary, max_len: int = 128, prepend_cls: bool = False):
        self.texts = frame['clean_review'].tolist()
        self.labels = frame['label'].astype('float32').tolist()
        self.raw_texts = frame['review'].tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.prepend_cls = prepend_cls

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, length, raw_len = self.vocab.encode(
            self.texts[idx], max_len=self.max_len, prepend_cls=self.prepend_cls
        )
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'raw_len': torch.tensor(raw_len, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'raw_text': self.raw_texts[idx],
            'clean_text': self.texts[idx]
        }


def make_dataloaders(max_len: int = 128, batch_size: int = 64, prepend_cls: bool = False):
    set_seed(SEED)
    train_ds = IMDbDataset(train_subset, vocab, max_len=max_len, prepend_cls=prepend_cls)
    val_ds = IMDbDataset(val_subset, vocab, max_len=max_len, prepend_cls=prepend_cls)
    test_ds = IMDbDataset(test_subset, vocab, max_len=max_len, prepend_cls=prepend_cls)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ==============================================================================
# 3. ĐỊNH NGHĨA MÔ HÌNH TRANSFORMER CLASSIFIER & LSTM CLASSIFIER
# ==============================================================================

class TransformerTextClassifier(nn.Module):
    """
    Mô hình phân loại văn bản sử dụng Transformer Encoder Block tự xây dựng (Mục 17, 21, 23, 24):
        Input IDs -> Embedding * sqrt(d_model) -> Positional Encoding
        -> TransformerBlock -> Pooling (Mean hoặc CLS) -> Linear Classifier -> Logit
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_len: int = 2048,
        dropout: float = 0.2,
        pooling: str = 'mean',
        use_positional_encoding: bool = True,
        use_pytorch_builtin_layer: bool = False
    ):
        super().__init__()
        assert pooling in ('mean', 'cls'), "pooling phải là 'mean' hoặc 'cls'"
        self.d_model = d_model
        self.n_heads = n_heads
        self.pooling = pooling
        self.use_positional_encoding = use_positional_encoding
        self.use_pytorch_builtin_layer = use_pytorch_builtin_layer

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Khởi tạo trọng số embedding theo chuẩn N(0, 1/sqrt(d_model)) để khi nhân sqrt(d_model)
        # phương sai bằng 1.0, cùng thang đo [-1, 1] với Positional Encoding và tránh bão hòa Softmax
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0.0)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.embed_dropout = nn.Dropout(dropout)

        if use_pytorch_builtin_layer:
            # Chỉ dùng để đối chiếu ở phần Application theo gợi ý Mục 17
            self.encoder_block = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation='relu', batch_first=True
            )
        else:
            # Core implementation bắt buộc của LAB 07
            self.encoder_block = TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
            )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None, return_attention: bool = False):
        # input_ids: (B, T)
        pad_mask = (input_ids != 0)  # (B, T), True ở token hợp lệ, False ở <PAD>
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # (B, T, d_model)

        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        else:
            x = self.embed_dropout(x)

        attn_weights = None
        if self.use_pytorch_builtin_layer:
            key_padding_mask = (input_ids == 0)  # PyTorch quy ước True là vị trí bị mask
            encoded = self.encoder_block(x, src_key_padding_mask=key_padding_mask)
        else:
            # mask cho MultiHeadSelfAttention: (B, 1, 1, T)
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            encoded, attn_weights = self.encoder_block(x, mask=attn_mask, return_attention=True)

        # Pooling: biến (B, T, d_model) -> (B, d_model)
        if self.pooling == 'mean':
            mask_float = pad_mask.unsqueeze(-1).float()  # (B, T, 1)
            sum_encoded = (encoded * mask_float).sum(dim=1)
            valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
            pooled = sum_encoded / valid_counts
        else:
            # First-token / [CLS]-style pooling: lấy vector tại vị trí đầu tiên t = 0
            pooled = encoded[:, 0, :]

        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        if return_attention:
            return logits, attn_weights
        return logits


class LSTMSentimentClassifier(nn.Module):
    """
    Model A (Mục 21 — W7.pdf):
        Embedding -> LSTM -> Classifier
    Giữ cùng d_model = 128, cùng bộ Classifier đầu ra để so sánh công bằng với Transformer.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        x = self.embed_dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        feat = hn.squeeze(0)  # (B, d_model)
        logits = self.classifier(feat).squeeze(-1)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# 4. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ CHUNG
# ==============================================================================

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 4,
    lr: float = 1e-3,
    model_name: str = "Model"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_state = None
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            x = batch['input_ids'].to(device)
            l = batch['length'].to(device)
            y = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(x, l)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).int()
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(y.detach().cpu().numpy().astype(int))

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Đánh giá trên tập Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'].to(device)
                l = batch['length'].to(device)
                y = batch['label'].to(device)
                logits = model(x, l)
                loss = criterion(logits, y)
                total_val_loss += loss.item() * len(y)
                preds = (torch.sigmoid(logits) >= 0.5).int()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy().astype(int))

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)

        history['train_loss'].append(round(avg_train_loss, 4))
        history['val_loss'].append(round(avg_val_loss, 4))
        history['train_acc'].append(round(train_acc, 4))
        history['val_acc'].append(round(val_acc, 4))

        print(
            f"  [{model_name}] Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.4f})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    training_time = time.time() - start_time
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Đánh giá trên tập Test
    model.eval()
    test_preds, test_probs, test_targets = [], [], []
    raw_texts, clean_texts = [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['input_ids'].to(device)
            l = batch['length'].to(device)
            y = batch['label'].to(device)
            logits = model(x, l)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            test_probs.extend(probs)
            test_preds.extend(preds)
            test_targets.extend(y.cpu().numpy().astype(int))
            raw_texts.extend(batch['raw_text'])
            clean_texts.extend(batch['clean_text'])

    acc = accuracy_score(test_targets, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average='binary', zero_division=0
    )

    print(
        f"  => [{model_name}] Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
        f"Recall: {rec:.4f} | F1: {f1:.4f} | Time: {training_time:.2f}s | Params: {count_parameters(model):,}\n"
    )

    return {
        'model': model,
        'model_name': model_name,
        'params': count_parameters(model),
        'training_time': round(training_time, 3),
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'history': history,
        'test_preds': np.array(test_preds),
        'test_probs': np.array(test_probs),
        'test_targets': np.array(test_targets),
        'raw_texts': raw_texts,
        'clean_texts': clean_texts
    }


# ==============================================================================
# 5. EXPERIMENT 2 — NUMBER OF ATTENTION HEADS (MỤC 19)
# ==============================================================================
print("=" * 75)
print("MỤC 19: EXPERIMENT 2 — NUMBER OF ATTENTION HEADS (d_model = 128)")
print("=" * 75)

train_loader_128, val_loader_128, test_loader_128 = make_dataloaders(max_len=128, batch_size=64)

exp2_heads_list = [1, 2, 4, 8]
exp2_records = []
trained_head_models = {}

for h in exp2_heads_list:
    set_seed(SEED)
    d_k = 128 // h
    print(f"[*] Huấn luyện Transformer với n_heads = {h} (d_k = {d_k})...")
    m_head = TransformerTextClassifier(
        vocab_size=len(vocab.word2idx),
        d_model=128,
        n_heads=h,
        d_ff=256,
        max_len=2048,
        dropout=0.2,
        pooling='mean'
    )
    res_h = train_and_evaluate(
        m_head, train_loader_128, val_loader_128, test_loader_128,
        epochs=4, lr=1e-3, model_name=f"Transformer (heads={h})"
    )
    trained_head_models[h] = res_h
    exp2_records.append({
        'Heads': h,
        'd_k': d_k,
        'Parameters': res_h['params'],
        'Training_Time_sec': res_h['training_time'],
        'Accuracy': res_h['accuracy'],
        'Precision': res_h['precision'],
        'Recall': res_h['recall'],
        'F1': res_h['f1']
    })


# ==============================================================================
# 6. EXPERIMENT 3 — ATTENTION HEAD DIVERSITY (MỤC 20)
# ==============================================================================
print("=" * 75)
print("MỤC 20: EXPERIMENT 3 — ATTENTION HEAD DIVERSITY (VISUALIZE 4 HEADS)")
print("=" * 75)

main_transformer_res = trained_head_models[4]
main_transformer = main_transformer_res['model']
main_transformer.eval()

div_sentence = "the movie was not good at first but the ending was truly great"
div_tokens = div_sentence.split()
div_ids = [vocab.word2idx.get(w, 1) for w in div_tokens]
div_tensor = torch.tensor([div_ids], dtype=torch.long).to(device)

with torch.no_grad():
    div_logit, div_attn = main_transformer(div_tensor, return_attention=True)
    # div_attn shape: (1, 4, T, T)
    div_attn_np = div_attn[0].cpu().numpy()  # (4, T, T)

head_stats = []
for h_idx in range(4):
    mat = div_attn_np[h_idx]
    # Entropy trung bình trên các hàng
    row_entropies = -np.sum(mat * np.log(mat + 1e-12), axis=-1)
    avg_entropy = float(np.mean(row_entropies))
    max_weight = float(np.max(mat))
    # Tìm cặp (Query -> Key) có trọng số lớn nhất
    q_i, k_j = np.unravel_index(np.argmax(mat), mat.shape)
    head_stats.append({
        'head': h_idx + 1,
        'avg_entropy': round(avg_entropy, 4),
        'max_weight': round(max_weight, 4),
        'strongest_pair': f"'{div_tokens[q_i]}' -> '{div_tokens[k_j]}' ({mat[q_i, k_j]:.3f})"
    })
    print(
        f"  Head {h_idx+1}: Avg Entropy = {avg_entropy:.4f} | Max Weight = {max_weight:.4f} | "
        f"Strongest link: '{div_tokens[q_i]}' -> '{div_tokens[k_j]}' ({mat[q_i, k_j]:.3f})"
    )

# Tính độ tương đồng Cosine giữa các ma trận Attention (trải phẳng) của 4 heads
pairwise_sim = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        v1 = div_attn_np[i].flatten()
        v2 = div_attn_np[j].flatten()
        pairwise_sim[i, j] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
print("  Ma trận Cosine Similarity giữa 4 Attention Heads:\n", np.round(pairwise_sim, 4))

# Vẽ attention_heads.png (2x2 grid cho 4 heads)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for h_idx, ax in enumerate(axes.flat):
    mat = div_attn_np[h_idx]
    im = ax.imshow(mat, cmap='viridis', vmin=0.0, vmax=max(0.35, float(np.max(div_attn_np))))
    ax.set_xticks(range(len(div_tokens)))
    ax.set_yticks(range(len(div_tokens)))
    ax.set_xticklabels(div_tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(div_tokens, fontsize=9)
    st = head_stats[h_idx]
    ax.set_title(
        f"Head {h_idx+1} (Entropy: {st['avg_entropy']:.2f}, Max: {st['max_weight']:.2f})\n"
        f"Peak: {st['strongest_pair']}",
        fontsize=10.5, fontweight='bold'
    )
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    for r in range(len(div_tokens)):
        for c in range(len(div_tokens)):
            val = mat[r, c]
            if val >= 0.12:
                ax.text(c, r, f"{val:.2f}", ha='center', va='center',
                        color='black' if val > 0.22 else 'white', fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    "Experiment 3 — Multi-Head Attention Diversity (4 Heads, d_model = 128, d_k = 32)\n"
    f"Sentence: \"{div_sentence}\"",
    fontsize=13, fontweight='bold', y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("attention_heads.png", dpi=300)
plt.close()
print("[+] Đã lưu: attention_heads.png\n")


# ==============================================================================
# 7. EXPERIMENT 4 — TRANSFORMER VS RNN/LSTM (MỤC 21)
# ==============================================================================
print("=" * 75)
print("MỤC 21: EXPERIMENT 4 — TRANSFORMER VS RNN/LSTM (CÙNG HYPERPARAMETERS)")
print("=" * 75)

set_seed(SEED)
print("[*] Huấn luyện Model A: LSTM Classifier (d_model = 128)...")
lstm_model = LSTMSentimentClassifier(vocab_size=len(vocab.word2idx), d_model=128, dropout=0.2)
lstm_res = train_and_evaluate(
    lstm_model, train_loader_128, val_loader_128, test_loader_128,
    epochs=4, lr=1e-3, model_name="LSTM (Model A)"
)

# Ngoài ra huấn luyện thêm PyTorch built-in TransformerEncoderLayer để đối chiếu ở phần Application (Mục 17)
set_seed(SEED)
print("[*] Huấn luyện Model B (PyTorch API đối chiếu): nn.TransformerEncoderLayer...")
pytorch_tf_model = TransformerTextClassifier(
    vocab_size=len(vocab.word2idx), d_model=128, n_heads=4, d_ff=256,
    max_len=2048, dropout=0.2, pooling='mean', use_pytorch_builtin_layer=True
)
pytorch_tf_res = train_and_evaluate(
    pytorch_tf_model, train_loader_128, val_loader_128, test_loader_128,
    epochs=4, lr=1e-3, model_name="PyTorch TransformerEncoderLayer"
)

exp4_comparison = [
    {
        'Model': 'LSTM (Model A)',
        'Parameters': lstm_res['params'],
        'Training_Time_sec': lstm_res['training_time'],
        'Accuracy': lstm_res['accuracy'],
        'Precision': lstm_res['precision'],
        'Recall': lstm_res['recall'],
        'F1': lstm_res['f1']
    },
    {
        'Model': 'Transformer (Custom Block - Model B)',
        'Parameters': main_transformer_res['params'],
        'Training_Time_sec': main_transformer_res['training_time'],
        'Accuracy': main_transformer_res['accuracy'],
        'Precision': main_transformer_res['precision'],
        'Recall': main_transformer_res['recall'],
        'F1': main_transformer_res['f1']
    },
    {
        'Model': 'Transformer (PyTorch API Reference)',
        'Parameters': pytorch_tf_res['params'],
        'Training_Time_sec': pytorch_tf_res['training_time'],
        'Accuracy': pytorch_tf_res['accuracy'],
        'Precision': pytorch_tf_res['precision'],
        'Recall': pytorch_tf_res['recall'],
        'F1': pytorch_tf_res['f1']
    }
]


# ==============================================================================
# 8. VẼ LEARNING CURVES (MỤC 29) & CONFUSION MATRIX (DELIVERABLES)
# ==============================================================================
# 8.1 Learning Curves (training loss, validation loss, validation accuracy)
epochs_range = list(range(1, 5))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Loss subplot
axes[0].plot(epochs_range, main_transformer_res['history']['train_loss'], 'o-', linewidth=2,
             color='#2980b9', label='Transformer — Train Loss')
axes[0].plot(epochs_range, main_transformer_res['history']['val_loss'], 's--', linewidth=2,
             color='#e74c3c', label='Transformer — Val Loss')
axes[0].plot(epochs_range, lstm_res['history']['train_loss'], '^:', linewidth=1.8,
             color='#27ae60', label='LSTM — Train Loss')
axes[0].plot(epochs_range, lstm_res['history']['val_loss'], 'd-.', linewidth=1.8,
             color='#8e44ad', label='LSTM — Val Loss')
axes[0].set_title("Training & Validation Loss across Epochs", fontsize=11.5, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCEWithLogits Loss")
axes[0].set_xticks(epochs_range)
axes[0].legend(fontsize=9.5)
axes[0].grid(True, linestyle='--', alpha=0.4)

# Accuracy subplot
axes[1].plot(epochs_range, main_transformer_res['history']['train_acc'], 'o-', linewidth=2,
             color='#2980b9', label='Transformer — Train Acc')
axes[1].plot(epochs_range, main_transformer_res['history']['val_acc'], 's--', linewidth=2,
             color='#e74c3c', label='Transformer — Val Acc')
axes[1].plot(epochs_range, lstm_res['history']['train_acc'], '^:', linewidth=1.8,
             color='#27ae60', label='LSTM — Train Acc')
axes[1].plot(epochs_range, lstm_res['history']['val_acc'], 'd-.', linewidth=1.8,
             color='#8e44ad', label='LSTM — Val Acc')
axes[1].set_title("Training & Validation Accuracy across Epochs", fontsize=11.5, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticks(epochs_range)
axes[1].legend(fontsize=9.5)
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.suptitle("LAB 07 — Learning Curves: Transformer Encoder vs LSTM on IMDb", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=300)
plt.close()
print("[+] Đã lưu: learning_curves.png")

# 8.2 Confusion Matrix
cm = confusion_matrix(main_transformer_res['test_targets'], main_transformer_res['test_preds'])
fig, ax = plt.subplots(figsize=(6.2, 5.2))
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=13, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=10.5)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=10.5)
ax.xaxis.set_ticks_position('bottom')
plt.title(
    f"Confusion Matrix — Transformer Text Classifier\n"
    f"Test Accuracy: {main_transformer_res['accuracy']:.4f} | F1-Score: {main_transformer_res['f1']:.4f}",
    pad=15, fontsize=11.5, fontweight='bold'
)
plt.xlabel("Predicted Sentiment", fontsize=10.5)
plt.ylabel("True Sentiment", fontsize=10.5)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("[+] Đã lưu: confusion_matrix.png\n")


# ==============================================================================
# 9. EXPERIMENT 5 — SEQUENCE LENGTH & O(T^2) COMPLEXITY (MỤC 22)
# ==============================================================================
print("=" * 75)
print("MỤC 22: EXPERIMENT 5 — SEQUENCE LENGTH (32, 64, 128, 256)")
print("=" * 75)

seq_lengths_exp5 = [32, 64, 128, 256]
exp5_records = []

# Tính độ dài trung bình thực tế của các bài review trong tập dữ liệu
all_token_counts = [len(t.split()) for t in test_subset['clean_review'].tolist()]

for max_l in seq_lengths_exp5:
    avg_retained = float(np.mean([min(c, max_l) for c in all_token_counts]))
    pct_retained = float(np.mean([min(c, max_l) / max(c, 1) * 100.0 for c in all_token_counts]))

    # Số phần tử trong ma trận Attention cho 1 mẫu (T^2) và cho 1 batch (B=64, n_heads=4, float32=4 bytes)
    t_squared = max_l * max_l
    batch_attn_elements = 64 * 4 * t_squared
    batch_attn_mb = (batch_attn_elements * 4) / (1024 * 1024)

    if max_l == 128:
        # Tái sử dụng kết quả đã train ở cấu hình chuẩn max_len = 128
        res_l = main_transformer_res
    else:
        tr_l, va_l, te_l = make_dataloaders(max_len=max_l, batch_size=64)
        set_seed(SEED)
        print(f"[*] Huấn luyện Transformer với max_length = {max_l}...")
        m_len = TransformerTextClassifier(
            vocab_size=len(vocab.word2idx), d_model=128, n_heads=4, d_ff=256,
            max_len=max(512, max_l + 16), dropout=0.2, pooling='mean'
        )
        res_l = train_and_evaluate(
            m_len, tr_l, va_l, te_l,
            epochs=4, lr=1e-3, model_name=f"Transformer (L={max_l})"
        )

    rec_item = {
        'Length': max_l,
        'Avg_Tokens_Retained': round(avg_retained, 1),
        'Pct_Tokens_Retained': round(pct_retained, 1),
        'T_Squared_Elements': t_squared,
        'Batch_Attn_Memory_MB': round(batch_attn_mb, 3),
        'Training_Time_sec': res_l['training_time'],
        'Accuracy': res_l['accuracy'],
        'Precision': res_l['precision'],
        'Recall': res_l['recall'],
        'F1': res_l['f1']
    }
    exp5_records.append(rec_item)
    print(
        f"  Length={max_l:3d} | Retained: {avg_retained:5.1f} tokens ({pct_retained:4.1f}%) | "
        f"T^2={t_squared:6,d} | Batch Attn Mem={batch_attn_mb:6.3f} MB | "
        f"Time={res_l['training_time']:6.2f}s | F1={res_l['f1']:.4f}"
    )

# Đo thêm thời gian Forward pass của Transformer vs LSTM ở chuỗi rất dài (T=512, 1024, 2048) để kiểm chứng Prediction 6
print("\n[*] Kiểm chứng tốc độ theo chiều dài chuỗi (Prediction 6 — Scaling Benchmark):")
scaling_benchmark = []
for T_bench in [64, 128, 256, 512, 1024]:
    dummy_ids = torch.randint(1, 5000, (16, T_bench), device=device)
    dummy_lens = torch.full((16,), T_bench, dtype=torch.long, device=device)
    with torch.no_grad():
        t0 = time.time()
        for _ in range(5):
            _ = main_transformer(dummy_ids, dummy_lens)
        tf_ms = (time.time() - t0) / 5.0 * 1000.0

        t0 = time.time()
        for _ in range(5):
            _ = lstm_model(dummy_ids, dummy_lens)
        lstm_ms = (time.time() - t0) / 5.0 * 1000.0
    scaling_benchmark.append({
        'T': T_bench,
        'Transformer_ms': round(tf_ms, 2),
        'LSTM_ms': round(lstm_ms, 2)
    })
    print(f"  T={T_bench:4d} (Batch=16) | Transformer: {tf_ms:7.2f} ms | LSTM: {lstm_ms:7.2f} ms")


# ==============================================================================
# 10. EXPERIMENT 6 — POOLING STRATEGY: MEAN VS CLS-STYLE (MỤC 25)
# ==============================================================================
print("\n" + "=" * 75)
print("MỤC 25: EXPERIMENT 6 — POOLING STRATEGY (MEAN VS CLS-STYLE)")
print("=" * 75)

tr_cls, va_cls, te_cls = make_dataloaders(max_len=128, batch_size=64, prepend_cls=True)
set_seed(SEED)
print("[*] Huấn luyện Transformer với First-token / [CLS]-style Pooling...")
cls_model = TransformerTextClassifier(
    vocab_size=len(vocab.word2idx), d_model=128, n_heads=4, d_ff=256,
    max_len=2048, dropout=0.2, pooling='cls'
)
cls_res = train_and_evaluate(
    cls_model, tr_cls, va_cls, te_cls,
    epochs=4, lr=1e-3, model_name="Transformer (CLS Pooling)"
)

exp6_records = [
    {
        'Pooling': 'Mean',
        'Accuracy': main_transformer_res['accuracy'],
        'Precision': main_transformer_res['precision'],
        'Recall': main_transformer_res['recall'],
        'F1': main_transformer_res['f1'],
        'Training_Time_sec': main_transformer_res['training_time']
    },
    {
        'Pooling': 'CLS-style',
        'Accuracy': cls_res['accuracy'],
        'Precision': cls_res['precision'],
        'Recall': cls_res['recall'],
        'F1': cls_res['f1'],
        'Training_Time_sec': cls_res['training_time']
    }
]


# ==============================================================================
# 11. APPLICATION — TRANSFORMER TEXT CLASSIFICATION (MỤC 23)
# ==============================================================================
print("=" * 75)
print("MỤC 23: APPLICATION — TRANSFORMER TEXT CLASSIFICATION (5 NHÓM CÂU)")
print("=" * 75)

application_test_cases = [
    {
        'category': 'Short sentence (Câu ngắn)',
        'text': "I really enjoyed this movie."
    },
    {
        'category': 'Long sentence (Câu dài)',
        'text': (
            "From the opening scene to the final credits, the director weaves an unforgettable "
            "story with brilliant cinematography, deeply moving performances from the entire cast, "
            "and a soundtrack that elevates every emotional moment into a true cinema classic."
        )
    },
    {
        'category': 'Negation (Phủ định)',
        'text': "I did not enjoy this movie at all, the plot was neither funny nor interesting."
    },
    {
        'category': 'Contrast (Tương phản chuyển ý)',
        'text': "I thought this film would be terrible and boring, but it turned out to be an excellent masterpiece."
    },
    {
        'category': 'Ambiguous sentiment (Cảm xúc mơ hồ / pha trộn)',
        'text': "The visual effects were stunning and ambitious, yet the story felt hollow, slow, and exhausting."
    }
]

app_predictions = []
main_transformer.eval()
for item in application_test_cases:
    cleaned = clean_text(item['text'])
    tokens = cleaned.split()
    ids, length, _ = vocab.encode(cleaned, max_len=max(len(tokens), 8))
    t_ids = torch.tensor([ids[:len(tokens)]], dtype=torch.long).to(device)
    with torch.no_grad():
        logit, attn = main_transformer(t_ids, return_attention=True)
        prob_pos = float(torch.sigmoid(logit).item())
        pred_label = "Positive" if prob_pos >= 0.5 else "Negative"
        # Tính trung bình attention nhận được của từng token trên 4 heads
        avg_attn_received = attn[0].mean(dim=(0, 1)).cpu().numpy()
        top_idx = np.argsort(avg_attn_received)[::-1][:3]
        top_tokens = [f"{tokens[i]} ({avg_attn_received[i]:.3f})" for i in top_idx if i < len(tokens)]

    record = {
        'category': item['category'],
        'text': item['text'],
        'predicted_label': pred_label,
        'probability_positive': round(prob_pos, 4),
        'top_attended_tokens': top_tokens
    }
    app_predictions.append(record)
    print(f"[{item['category']}]")
    print(f"  Input      : \"{item['text']}\"")
    print(f"  Output     : {pred_label} (Probability Positive: {prob_pos:.4f})")
    print(f"  Top Tokens : {', '.join(top_tokens)}\n")


# ==============================================================================
# 12. TRÍCH XUẤT 3 CA ĐÚNG & 3 CA SAI CHO ERROR ANALYSIS (MỤC 26)
# ==============================================================================
print("=" * 75)
print("MỤC 26: TRÍCH XUẤT 3 CA ĐÚNG & 3 CA SAI KÈM PHÂN TÍCH ATTENTION")
print("=" * 75)

preds_arr = main_transformer_res['test_preds']
targets_arr = main_transformer_res['test_targets']
probs_arr = main_transformer_res['test_probs']

correct_indices = np.where(preds_arr == targets_arr)[0]
incorrect_indices = np.where(preds_arr != targets_arr)[0]
print(f"[*] Tổng số mẫu Test: {len(targets_arr)} | Đúng: {len(correct_indices)} | Sai: {len(incorrect_indices)}")

def inspect_sample(idx: int):
    raw_txt = main_transformer_res['raw_texts'][idx]
    clean_txt = main_transformer_res['clean_texts'][idx]
    tokens = clean_txt.split()[:128]
    ids, length, raw_len = vocab.encode(clean_txt, max_len=128)
    t_ids = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logit, attn = main_transformer(t_ids, return_attention=True)
        # attn: (1, 4, 128, 128)
        valid_len = min(len(tokens), 128)
        attn_valid = attn[0, :, :valid_len, :valid_len].cpu().numpy()  # (4, valid_len, valid_len)
        mean_received = attn_valid.mean(axis=(0, 1))  # (valid_len,)
        top_k_idx = np.argsort(mean_received)[::-1][:6]
        top_tokens = [f"{tokens[i]} ({mean_received[i]:.3f})" for i in top_k_idx]

        per_head_top = {}
        for h in range(4):
            h_rec = attn_valid[h].mean(axis=0)
            h_top = np.argsort(h_rec)[::-1][:3]
            per_head_top[f"Head_{h+1}"] = [f"{tokens[i]} ({h_rec[i]:.3f})" for i in h_top]

    return {
        'sample_id': int(idx),
        'true_label': 'Positive' if targets_arr[idx] == 1 else 'Negative',
        'pred_label': 'Positive' if preds_arr[idx] == 1 else 'Negative',
        'prob_positive': round(float(probs_arr[idx]), 4),
        'raw_len': int(raw_len),
        'retained_len': int(length),
        'truncated': bool(raw_len > 128),
        'top_attended_tokens_overall': top_tokens,
        'per_head_top_tokens': per_head_top,
        'raw_text': raw_txt
    }

selected_correct_ids = [int(i) for i in correct_indices[:5]]
selected_incorrect_ids = [int(i) for i in incorrect_indices[:8]]

correct_cases_info = [inspect_sample(i) for i in selected_correct_ids]
incorrect_cases_info = [inspect_sample(i) for i in selected_incorrect_ids]

for c in correct_cases_info[:3]:
    print(f"  [CORRECT #{c['sample_id']}] True={c['true_label']} | Pred={c['pred_label']} (Prob={c['prob_positive']})")
    print(f"    Len={c['raw_len']} (Truncated={c['truncated']}) | Top Attended: {', '.join(c['top_attended_tokens_overall'])}")
    print(f"    Excerpt: {c['raw_text'][:140]}...\n")

for ic in incorrect_cases_info[:5]:
    print(f"  [INCORRECT #{ic['sample_id']}] True={ic['true_label']} | Pred={ic['pred_label']} (Prob={ic['prob_positive']})")
    print(f"    Len={ic['raw_len']} (Truncated={ic['truncated']}) | Top Attended: {', '.join(ic['top_attended_tokens_overall'])}")
    print(f"    Excerpt: {ic['raw_text'][:140]}...\n")


# ==============================================================================
# 13. XUẤT FILE results.csv & lab07_summary.json
# ==============================================================================
all_csv_rows = []

# Nhóm 1: Experiment 2 (Number of Attention Heads)
for r in exp2_records:
    all_csv_rows.append({
        'Experiment': 'Exp2_Attention_Heads',
        'Configuration': f"Transformer (n_heads={r['Heads']}, d_k={r['d_k']})",
        'd_model': 128,
        'n_heads': r['Heads'],
        'd_k': r['d_k'],
        'max_length': 128,
        'Pooling': 'Mean',
        'Parameters': r['Parameters'],
        'Training_Time_sec': r['Training_Time_sec'],
        'Accuracy': r['Accuracy'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1': r['F1']
    })

# Nhóm 2: Experiment 4 (LSTM vs Transformer)
all_csv_rows.append({
    'Experiment': 'Exp4_LSTM_vs_Transformer',
    'Configuration': 'LSTM Baseline (Model A)',
    'd_model': 128,
    'n_heads': 0,
    'd_k': 0,
    'max_length': 128,
    'Pooling': 'Final_Hidden',
    'Parameters': lstm_res['params'],
    'Training_Time_sec': lstm_res['training_time'],
    'Accuracy': lstm_res['accuracy'],
    'Precision': lstm_res['precision'],
    'Recall': lstm_res['recall'],
    'F1': lstm_res['f1']
})
all_csv_rows.append({
    'Experiment': 'Exp4_LSTM_vs_Transformer',
    'Configuration': 'Transformer Custom Block (Model B)',
    'd_model': 128,
    'n_heads': 4,
    'd_k': 32,
    'max_length': 128,
    'Pooling': 'Mean',
    'Parameters': main_transformer_res['params'],
    'Training_Time_sec': main_transformer_res['training_time'],
    'Accuracy': main_transformer_res['accuracy'],
    'Precision': main_transformer_res['precision'],
    'Recall': main_transformer_res['recall'],
    'F1': main_transformer_res['f1']
})
all_csv_rows.append({
    'Experiment': 'Exp4_LSTM_vs_Transformer',
    'Configuration': 'Transformer PyTorch API Reference',
    'd_model': 128,
    'n_heads': 4,
    'd_k': 32,
    'max_length': 128,
    'Pooling': 'Mean',
    'Parameters': pytorch_tf_res['params'],
    'Training_Time_sec': pytorch_tf_res['training_time'],
    'Accuracy': pytorch_tf_res['accuracy'],
    'Precision': pytorch_tf_res['precision'],
    'Recall': pytorch_tf_res['recall'],
    'F1': pytorch_tf_res['f1']
})

# Nhóm 3: Experiment 5 (Sequence Length)
for r in exp5_records:
    all_csv_rows.append({
        'Experiment': 'Exp5_Sequence_Length',
        'Configuration': f"Transformer (max_length={r['Length']})",
        'd_model': 128,
        'n_heads': 4,
        'd_k': 32,
        'max_length': r['Length'],
        'Pooling': 'Mean',
        'Parameters': main_transformer_res['params'],
        'Training_Time_sec': r['Training_Time_sec'],
        'Accuracy': r['Accuracy'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1': r['F1']
    })

# Nhóm 4: Experiment 6 (Pooling Strategy)
for r in exp6_records:
    all_csv_rows.append({
        'Experiment': 'Exp6_Pooling_Strategy',
        'Configuration': f"Transformer ({r['Pooling']} Pooling)",
        'd_model': 128,
        'n_heads': 4,
        'd_k': 32,
        'max_length': 128,
        'Pooling': r['Pooling'],
        'Parameters': main_transformer_res['params'],
        'Training_Time_sec': r['Training_Time_sec'],
        'Accuracy': r['Accuracy'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1': r['F1']
    })

results_df = pd.DataFrame(all_csv_rows)
results_df.to_csv("results.csv", index=False)
print("[+] Đã lưu bảng kết quả định lượng vào: results.csv")
print(results_df.to_string(index=False))

summary_payload = {
    'exp2_attention_heads': exp2_records,
    'exp3_head_diversity': {
        'sentence': div_sentence,
        'head_stats': head_stats,
        'pairwise_cosine_similarity': np.round(pairwise_sim, 4).tolist()
    },
    'exp4_lstm_vs_transformer': exp4_comparison,
    'exp5_sequence_length': exp5_records,
    'scaling_benchmark': scaling_benchmark,
    'exp6_pooling': exp6_records,
    'learning_curves': {
        'transformer': main_transformer_res['history'],
        'lstm': lstm_res['history']
    },
    'confusion_matrix': cm.tolist(),
    'application_predictions': app_predictions,
    'error_analysis': {
        'correct_cases': correct_cases_info,
        'incorrect_cases': incorrect_cases_info
    }
}

with open("lab07_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2, ensure_ascii=False)
print("[+] Đã lưu chi tiết toàn bộ thực nghiệm vào: lab07_summary.json")

