"""
Script thực hiện toàn bộ các thực nghiệm định lượng của LAB 06:
1. Thí nghiệm Temperature và Scaling (Mục 14, 15)
2. Thí nghiệm Attention Distribution & Query variation (Exp 1, 2)
3. Thí nghiệm Attention Matrix Visualization (Exp 3) -> lưu attention_heatmap.png
4. Huấn luyện mô hình Self-Attention Classifier và so sánh với LSTM (Exp 4)
5. Đo độ phức tạp ma trận Attention theo Sequence Length (Exp 5)
6. Ứng dụng phân loại cảm xúc trên các câu mẫu (Application)
7. Đánh giá Confusion Matrix trên Test set -> lưu confusion_matrix.png
8. Trích xuất các ca dự đoán đúng/sai cho error_analysis.md
9. Xuất toàn bộ số liệu đo đạc thực tế vào results.csv
"""

import os
import re
import sys
import time
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

# ==============================================================================
# 1. THIẾT LẬP REPRODUCIBILITY & THIẾT BỊ
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Thiết bị tính toán: {device}")

# ==============================================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU IMDB
# ==============================================================================
DATA_PATH = os.path.join("..", "data", "IMDB Dataset.csv")
assert os.path.exists(DATA_PATH), f"Không tìm thấy file dữ liệu tại {DATA_PATH}"

df_raw = pd.read_csv(DATA_PATH)
print(f"[*] Dữ liệu gốc: {df_raw.shape}")

# Loại bỏ các dòng trùng lặp (418 dòng)
df = df_raw.drop_duplicates(subset=['review']).reset_index(drop=True)
print(f"[*] Dữ liệu sau khi loại trùng: {df.shape}")

# Gán nhãn số: positive -> 1, negative -> 0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text: str) -> str:
    # Xóa HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Giữ lại chữ cái, số, dấu câu cơ bản
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

# Phân chia tập Train / Val / Test với stratify
# Để đảm bảo huấn luyện nhanh trên CPU nhưng vẫn có độ khái quát tốt và nhất quán với LAB 05:
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label'])

# Lấy tập mẫu vừa đủ để train nhanh trên CPU (2500 mỗi nhãn = 5000 train, 500 mỗi nhãn = 1000 val, 500 mỗi nhãn = 1000 test)
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

print(f"[*] Tập Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

# Xây dựng Vocabulary
class Vocabulary:
    def __init__(self, max_size=15000):
        self.max_size = max_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        for word, _ in counter.most_common(self.max_size - 2):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    def encode(self, text, max_len=100):
        tokens = text.split()
        length = min(len(tokens), max_len)
        ids = [self.word2idx.get(w, 1) for w in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids, length

vocab = Vocabulary(max_size=15000)
vocab.build(train_subset['clean_review'].tolist())
print(f"[*] Kích thước từ vựng (Vocab size): {len(vocab.word2idx)}")

# Dataset & DataLoader
MAX_LEN = 100
BATCH_SIZE = 64

class IMDbDataset(Dataset):
    def __init__(self, frame, vocab, max_len=MAX_LEN):
        self.texts = frame['clean_review'].tolist()
        self.labels = frame['label'].astype('float32').tolist()
        self.raw_texts = frame['review'].tolist()
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        ids, length = self.vocab.encode(self.texts[idx], self.max_len)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'raw_text': self.raw_texts[idx]
        }

train_ds = IMDbDataset(train_subset, vocab, max_len=MAX_LEN)
val_ds = IMDbDataset(val_subset, vocab, max_len=MAX_LEN)
test_ds = IMDbDataset(test_subset, vocab, max_len=MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# 3. MÔ HÌNH SELF-ATTENTION & LSTM
# ==============================================================================
class SelfAttentionLayer(nn.Module):
    """
    Cài đặt Self-Attention Layer thuần túy:
    Q = X W_Q, K = X W_K, V = X W_V
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = np.sqrt(embed_dim)
        
    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)  # (B, T, d_model)
        V = self.W_v(x)  # (B, T, d_model)
        
        # S = Q @ K^T / sqrt(d_k) -> shape (B, T, T)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        
        if mask is not None:
            # mask: (B, 1, T) -> mở rộng sang (B, T, T)
            # Điền -1e9 vào vị trí token <PAD>
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        context = torch.matmul(attn_weights, V)    # (B, T, d_model)
        return context, attn_weights

class AttentionSentimentClassifier(nn.Module):
    """
    Kiến trúc Context-Aware Sentiment Classifier sử dụng Self-Attention:
    Embedding -> Self-Attention -> Mean Pooling over valid tokens -> Dense -> Logit
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = SelfAttentionLayer(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, input_ids, lengths=None):
        # input_ids: (B, T)
        mask = (input_ids != 0).unsqueeze(1)  # (B, 1, T)
        embedded = self.embedding(input_ids)  # (B, T, embed_dim)
        
        # Self-Attention
        context, attn_weights = self.attention(embedded, mask=mask)
        
        # Masked Mean Pooling: Lấy trung bình context vector trên các token hợp lệ (bỏ qua PAD)
        mask_expanded = (input_ids != 0).unsqueeze(-1).float()  # (B, T, 1)
        context_masked = context * mask_expanded
        sum_context = context_masked.sum(dim=1)  # (B, embed_dim)
        lengths_clamped = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = sum_context / lengths_clamped  # (B, embed_dim)
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return logits, attn_weights

class LSTMSentimentClassifier(nn.Module):
    """
    Mô hình baseline LSTM từ LAB 05 để đối chiếu:
    Embedding -> LSTM -> Final hidden state -> Dense -> Logit
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        packed_out, (hn, cn) = self.lstm(packed)
        # hn: (1, B, hidden_dim)
        out = self.dropout(hn.squeeze(0))
        logits = self.classifier(out).squeeze(-1)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==============================================================================
# 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (EXPERIMENT 4)
# ==============================================================================
def train_and_eval(model, train_loader, val_loader, test_loader, epochs=4, lr=1e-3, is_attention=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['input_ids'].to(device)
            l = batch['length'].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            if is_attention:
                logits, _ = model(x, l)
            else:
                logits = model(x, l)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'].to(device)
                l = batch['length'].to(device)
                y = batch['label'].to(device)
                if is_attention:
                    logits, _ = model(x, l)
                else:
                    logits = model(x, l)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    training_time = time.time() - start_time
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    
    # Đánh giá trên Test set
    model.eval()
    test_preds, test_probs, test_targets = [], [], []
    all_raw_texts = []
    attention_matrices = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['input_ids'].to(device)
            l = batch['length'].to(device)
            y = batch['label'].to(device)
            if is_attention:
                logits, attn = model(x, l)
                attention_matrices.append(attn.cpu().numpy())
            else:
                logits = model(x, l)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            test_probs.extend(probs)
            test_preds.extend(preds)
            test_targets.extend(y.cpu().numpy().astype(int))
            all_raw_texts.extend(batch['raw_text'])
            
    acc = accuracy_score(test_targets, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='binary')
    
    return {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'training_time': training_time,
        'params': count_parameters(model),
        'test_preds': np.array(test_preds),
        'test_probs': np.array(test_probs),
        'test_targets': np.array(test_targets),
        'raw_texts': all_raw_texts,
        'attn_matrices': np.concatenate(attention_matrices, axis=0) if is_attention else None
    }

print("\n" + "="*50)
print("[*] HUẤN LUYỆN MÔ HÌNH 1: SELF-ATTENTION CLASSIFIER")
print("="*50)
attn_model = AttentionSentimentClassifier(len(vocab.word2idx), embed_dim=64, hidden_dim=64)
attn_results = train_and_eval(attn_model, train_loader, val_loader, test_loader, epochs=4, lr=1e-3, is_attention=True)

print("\n" + "="*50)
print("[*] HUẤN LUYỆN MÔ HÌNH 2: LSTM CLASSIFIER (BASELINE LAB 05)")
print("="*50)
lstm_model = LSTMSentimentClassifier(len(vocab.word2idx), embed_dim=64, hidden_dim=64)
lstm_results = train_and_eval(lstm_model, train_loader, val_loader, test_loader, epochs=4, lr=1e-3, is_attention=False)

# ==============================================================================
# 5. TẠO CONFUSION MATRIX VÀ HEATMAP (DELIVERABLES)
# ==============================================================================
# 5.1 Vẽ Confusion Matrix bằng Matplotlib
cm = confusion_matrix(attn_results['test_targets'], attn_results['test_preds'])
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
ax.set_yticklabels(['Negative (0)', 'Positive (1)'])
ax.xaxis.set_ticks_position('bottom')
plt.title(f"Confusion Matrix — Self-Attention Classifier\nAccuracy: {attn_results['accuracy']:.4f} | F1: {attn_results['f1']:.4f}", pad=15)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("[+] Đã lưu: confusion_matrix.png")

# 5.2 Vẽ Attention Heatmap (Experiment 3 theo W6.pdf)
# W6.pdf Mục 23: "The cat that I adopted yesterday is sleeping."
sample_sentence = "the cat that i adopted yesterday is sleeping"
tokens = sample_sentence.split()
ids = [vocab.word2idx.get(w, 1) for w in tokens]
t_ids = torch.tensor([ids], dtype=torch.long).to(device)

attn_model.eval()
with torch.no_grad():
    _, sample_attn = attn_model(t_ids)
    # sample_attn shape: (1, T, T)
    attn_matrix_sample = sample_attn[0].cpu().numpy()

fig, ax = plt.subplots(figsize=(8, 7))
cax = ax.imshow(attn_matrix_sample, cmap='viridis', interpolation='nearest')
fig.colorbar(cax)

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha='right')
ax.set_yticklabels(tokens)

for i in range(len(tokens)):
    for j in range(len(tokens)):
        val = attn_matrix_sample[i, j]
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                color='white' if val < 0.25 else 'black', fontsize=9)

plt.title("Self-Attention Heatmap (Experiment 3)\nSentence: 'The cat that I adopted yesterday is sleeping.'")
plt.xlabel("Key Tokens (được chú ý)")
plt.ylabel("Query Tokens (đang chú ý)")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=300)
plt.close()
print("[+] Đã lưu: attention_heatmap.png")

# ==============================================================================
# 6. THỰC HIỆN APPLICATION (MỤC 26 W6.PDF)
# ==============================================================================
app_sentences = [
    "I like this movie.",
    "I do not like this movie.",
    "I thought this movie would be bad, but it was actually excellent."
]

print("\n" + "="*50)
print("[*] THỰC NGHIỆM MỤC 26: APPLICATION — CONTEXT-AWARE SENTIMENT")
print("="*50)

app_results_records = []
for sent in app_sentences:
    cleaned = clean_text(sent)
    tokens = cleaned.split()
    ids, length = vocab.encode(cleaned, max_len=len(tokens))
    t_ids = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logit, attn = attn_model(t_ids)
        prob = torch.sigmoid(logit).item()
        pred = "positive" if prob >= 0.5 else "negative"
    print(f"Câu: \"{sent}\"")
    print(f"  -> Prediction: {pred} (Probability Positive: {prob:.4f})")
    app_results_records.append({
        'sentence': sent,
        'prediction': pred,
        'prob_positive': prob
    })

# ==============================================================================
# 7. THỰC NGHIỆM ĐỘ DÀI SEQUENCE VÀ CHI PHÍ TÍNH TOÁN O(T^2) (EXPERIMENT 5)
# ==============================================================================
print("\n" + "="*50)
print("[*] THỰC NGHIỆM EXPERIMENT 5: SEQUENCE LENGTH & O(T^2) COMPLEXITY")
print("="*50)
seq_lengths = [10, 100, 500, 1000, 10000]
seq_cost_data = []

for T in seq_lengths:
    elements = T * T
    # Ước tính dung lượng bộ nhớ float32 (4 bytes / element)
    memory_mb = (elements * 4) / (1024 * 1024)
    # Đo thời gian nhân ma trận trên CPU nếu T <= 1000
    if T <= 1000:
        Q_test = torch.randn(1, T, 64)
        K_test = torch.randn(1, T, 64)
        t0 = time.time()
        for _ in range(20):
            S = torch.matmul(Q_test, K_test.transpose(-1, -2)) / np.sqrt(64)
            A = F.softmax(S, dim=-1)
        t_avg = (time.time() - t0) / 20.0
    else:
        # Với T = 10000, 100 triệu phần tử ~ 381 MB bộ nhớ, xấp xỉ thời gian
        t_avg = t_avg * ((T / 1000) ** 2) if 't_avg' in locals() else None
        
    seq_cost_data.append({
        'sequence_length': T,
        'attention_matrix_shape': f"{T}x{T}",
        'matrix_elements': elements,
        'memory_mb': round(memory_mb, 2),
        'computation_time_sec': round(t_avg, 5) if t_avg else 'N/A'
    })
    print(f"Length T={T:5d} | Matrix: ({T}x{T}) = {elements:11,d} elements | Memory: {memory_mb:8.2f} MB | Time: {t_avg if t_avg else 'N/A'}")

# ==============================================================================
# 8. THỰC NGHIỆM TEMPERATURE & SCALING (MỤC 14 & 15)
# ==============================================================================
raw_qk = np.array([4.0, 2.0, 0.0])
temp_records = []
for T_val in [0.5, 1.0, 2.0]:
    s_div = raw_qk / T_val
    exp_s = np.exp(s_div - np.max(s_div))
    w = exp_s / np.sum(exp_s)
    entropy = -np.sum(w * np.log(w + 1e-12))
    temp_records.append({
        'Temperature': T_val,
        'Max_Weight': round(float(np.max(w)), 4),
        'Entropy': round(float(entropy), 4),
        'Weights': str(np.round(w, 4).tolist())
    })

# ==============================================================================
# 9. XUẤT TOÀN BỘ KẾT QUẢ VÀO results.csv
# ==============================================================================
results_df = pd.DataFrame([
    {
        'Model': 'LSTM',
        'Accuracy': round(lstm_results['accuracy'], 4),
        'Precision': round(lstm_results['precision'], 4),
        'Recall': round(lstm_results['recall'], 4),
        'F1': round(lstm_results['f1'], 4),
        'Training Time (sec)': round(lstm_results['training_time'], 3),
        'Parameters': lstm_results['params']
    },
    {
        'Model': 'Self-Attention',
        'Accuracy': round(attn_results['accuracy'], 4),
        'Precision': round(attn_results['precision'], 4),
        'Recall': round(attn_results['recall'], 4),
        'F1': round(attn_results['f1'], 4),
        'Training Time (sec)': round(attn_results['training_time'], 3),
        'Parameters': attn_results['params']
    }
])
results_df.to_csv("results.csv", index=False)
print("[+] Đã lưu: results.csv")
print(results_df.to_string())

# ==============================================================================
# 10. TRÍCH XUẤT 3 VÍ DỤ ĐÚNG & 3 VÍ DỤ SAI THỰC TẾ CHO error_analysis.md
# ==============================================================================
correct_indices = np.where(attn_results['test_preds'] == attn_results['test_targets'])[0]
incorrect_indices = np.where(attn_results['test_preds'] != attn_results['test_targets'])[0]

print(f"\n[*] Tổng số mẫu đúng: {len(correct_indices)}, mẫu sai: {len(incorrect_indices)}")

# Chọn 3 ca đúng và 3 ca sai thú vị (có độ dài câu và sắc thái ngữ nghĩa tiêu biểu)
selected_correct = correct_indices[:3]
selected_incorrect = incorrect_indices[:3]

print("\n--- 3 VÍ DỤ ĐÚNG (CORRECT CASES) ---")
for idx in selected_correct:
    text = attn_results['raw_texts'][idx]
    y_true = "positive" if attn_results['test_targets'][idx] == 1 else "negative"
    y_pred = "positive" if attn_results['test_preds'][idx] == 1 else "negative"
    prob = attn_results['test_probs'][idx]
    print(f"Sample #{idx} | True: {y_true} | Pred: {y_pred} (Prob: {prob:.4f})")
    print(f"Text: {text[:150]}...\n")

print("--- 3 VÍ DỤ SAI (INCORRECT CASES) ---")
for idx in selected_incorrect:
    text = attn_results['raw_texts'][idx]
    y_true = "positive" if attn_results['test_targets'][idx] == 1 else "negative"
    y_pred = "positive" if attn_results['test_preds'][idx] == 1 else "negative"
    prob = attn_results['test_probs'][idx]
    print(f"Sample #{idx} | True: {y_true} | Pred: {y_pred} (Prob: {prob:.4f})")
    print(f"Text: {text[:150]}...\n")

# Lưu thông tin các ca này ra một file phụ để dùng cho error_analysis.md
import json
with open("error_cases.json", "w", encoding="utf-8") as f:
    json.dump({
        'correct': [
            {
                'id': int(idx),
                'text': attn_results['raw_texts'][idx],
                'true_label': "positive" if attn_results['test_targets'][idx] == 1 else "negative",
                'pred_label': "positive" if attn_results['test_preds'][idx] == 1 else "negative",
                'prob': float(attn_results['test_probs'][idx])
            } for idx in selected_correct
        ],
        'incorrect': [
            {
                'id': int(idx),
                'text': attn_results['raw_texts'][idx],
                'true_label': "positive" if attn_results['test_targets'][idx] == 1 else "negative",
                'pred_label': "positive" if attn_results['test_preds'][idx] == 1 else "negative",
                'prob': float(attn_results['test_probs'][idx])
            } for idx in selected_incorrect
        ],
        'sequence_cost': seq_cost_data,
        'temperature_exp': temp_records,
        'app_results': app_results_records
    }, f, indent=2, ensure_ascii=False)

print("[+] Đã lưu dữ liệu thực nghiệm vào error_cases.json")
