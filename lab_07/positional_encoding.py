"""
NLP LAB 07 — TRANSFORMER ARCHITECTURE
Chủ đề: From Attention to Transformer

File: positional_encoding.py
Mục đích:
    - Cài đặt Sinusoidal Positional Encoding (Vaswani et al., 2017) bằng NumPy và PyTorch.
    - Kiểm chứng bài tập tính tay Mục 10 (d = 4, pos = 0, 1, 2).
    - Thực hiện Experiment 1 (Mục 11): So sánh biểu diễn câu ["I", "love", "NLP"] và ["NLP", "love", "I"]
      khi không có và khi có Positional Encoding.
    - Thực hiện Permutation Experiment (Mục 12): Chứng minh thực nghiệm với
      S1 = "dog bites man" và S2 = "man bites dog".
    - Trực quan hóa ma trận Positional Encoding và lưu thành file positional_encoding.png.
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Đảm bảo mã hóa UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def compute_positional_encoding_numpy(max_len: int, d_model: int) -> np.ndarray:
    """
    Tính toán ma trận Sinusoidal Positional Encoding bằng NumPy.

    Công thức (Mục 9 — W7.pdf):
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        max_len (int): Độ dài chuỗi tối đa (số lượng vị trí pos = 0, 1, ..., max_len - 1).
        d_model (int): Số chiều vector biểu diễn (phải là số chẵn).

    Returns:
        np.ndarray: Ma trận PE có kích thước (max_len, d_model).
    """
    assert d_model % 2 == 0, f"d_model ({d_model}) phải là số chẵn!"
    pe = np.zeros((max_len, d_model), dtype=np.float64)
    position = np.arange(0, max_len, dtype=np.float64)[:, np.newaxis]  # (max_len, 1)
    div_term = np.power(10000.0, np.arange(0, d_model, 2, dtype=np.float64) / d_model)  # (d_model/2,)

    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)
    return pe


class PositionalEncoding(nn.Module):
    """
    Module Sinusoidal Positional Encoding trong PyTorch.
    Cộng trực tiếp thông tin vị trí vào Input Embeddings:
        X' = X + PE[:T, :]
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Đăng ký buffer kích thước (1, max_len, d_model) để tự động chuyển theo device (CPU/GPU)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input embeddings có shape (B, T, d_model)
        Returns:
            torch.Tensor: Embeddings đã cộng Positional Encoding, shape (B, T, d_model)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


def run_section_10_calculation():
    """Kiểm chứng bài tập tính tay Mục 10: d = 4, pos = 0, 1, 2."""
    print("=" * 75)
    print("MỤC 10: BÀI TẬP TÍNH POSITIONAL ENCODING (d = 4, pos = 0, 1, 2)")
    print("=" * 75)
    d = 4
    positions = [0, 1, 2]
    pe = compute_positional_encoding_numpy(max_len=3, d_model=d)

    # Với d = 4:
    # i = 0 -> 2i = 0, 2i+1 = 1 -> mẫu số: 10000^(0/4) = 10000^0 = 1
    # i = 1 -> 2i = 2, 2i+1 = 3 -> mẫu số: 10000^(2/4) = 10000^0.5 = 100
    for pos in positions:
        expected = np.array([
            math.sin(pos / 1.0),
            math.cos(pos / 1.0),
            math.sin(pos / 100.0),
            math.cos(pos / 100.0)
        ])
        assert np.allclose(pe[pos], expected, atol=1e-7), f"Sai lệch tại pos={pos}"
        print(f"pos = {pos}:")
        print(f"  Công thức: [sin({pos}/1), cos({pos}/1), sin({pos}/100), cos({pos}/100)]")
        print(f"  Giá trị  : {np.round(pe[pos], 6)}")
    print("-> Nhận xét: Mỗi vị trí pos = 0, 1, 2 có một vector biểu diễn vị trí hoàn toàn khác nhau.")
    print("-> Kiểm chứng Mục 10: PASSED!\n")
    return pe


def simple_self_attention_numpy(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
    """Hàm Self-Attention đơn giản phục vụ thực nghiệm đối chiếu vị trí."""
    d_k = W_K.shape[-1]
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = (Q @ K.T) / math.sqrt(d_k)
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=-1, keepdims=True)
    out = attn @ V
    return out, attn


def run_experiment_1_and_permutation():
    """
    Thực hiện:
    - Experiment 1 (Mục 11): ["I", "love", "NLP"] vs ["NLP", "love", "I"]
    - Permutation Experiment (Mục 12): S1 = "dog bites man" vs S2 = "man bites dog"
    """
    print("=" * 75)
    print("MỤC 11: EXPERIMENT 1 — POSITIONAL ENCODING (['I', 'love', 'NLP'])")
    print("=" * 75)
    np.random.seed(42)
    d_model = 4
    vocab_exp1 = {"I": 0, "love": 1, "NLP": 2}
    embed_table_1 = np.array([
        [1.0, 0.2, -0.5, 0.8],   # "I"
        [0.4, 1.2,  0.3, -0.2],  # "love"
        [-0.6, 0.5, 1.5, 0.1]    # "NLP"
    ])
    W_Q = np.array([[0.5, -0.2, 0.1, 0.4],
                    [0.1,  0.6, -0.3, 0.2],
                    [-0.4, 0.3, 0.7, -0.1],
                    [0.2, -0.5, 0.2, 0.8]])
    W_K = np.array([[0.3,  0.4, -0.2, 0.1],
                    [-0.1, 0.5,  0.4, -0.3],
                    [0.6, -0.2,  0.5, 0.2],
                    [0.2,  0.1, -0.4, 0.6]])
    W_V = np.array([[0.7, -0.1, 0.2, 0.3],
                    [0.2,  0.8, -0.2, 0.1],
                    [-0.3, 0.2, 0.9, -0.2],
                    [0.1, -0.3, 0.1, 0.7]])

    seq_a = ["I", "love", "NLP"]
    seq_b = ["NLP", "love", "I"]

    X_a = np.array([embed_table_1[vocab_exp1[w]] for w in seq_a])
    X_b = np.array([embed_table_1[vocab_exp1[w]] for w in seq_b])
    PE_3 = compute_positional_encoding_numpy(max_len=3, d_model=d_model)

    # Không có PE
    out_a_no_pe, _ = simple_self_attention_numpy(X_a, W_Q, W_K, W_V)
    out_b_no_pe, _ = simple_self_attention_numpy(X_b, W_Q, W_K, W_V)

    # Có PE
    out_a_with_pe, _ = simple_self_attention_numpy(X_a + PE_3, W_Q, W_K, W_V)
    out_b_with_pe, _ = simple_self_attention_numpy(X_b + PE_3, W_Q, W_K, W_V)

    print("1. Không có Positional Encoding:")
    print(f"   Biểu diễn token 'love' (vị trí giữa) trong 'I love NLP' : {np.round(out_a_no_pe[1], 6)}")
    print(f"   Biểu diễn token 'love' (vị trí giữa) trong 'NLP love I' : {np.round(out_b_no_pe[1], 6)}")
    print(f"   Mean-pooled vector 'I love NLP' : {np.round(out_a_no_pe.mean(axis=0), 6)}")
    print(f"   Mean-pooled vector 'NLP love I' : {np.round(out_b_no_pe.mean(axis=0), 6)}")
    diff_no_pe = np.linalg.norm(out_a_no_pe.mean(axis=0) - out_b_no_pe.mean(axis=0))
    print(f"   -> Khoảng cách L2 giữa 2 vector toàn câu (không PE): {diff_no_pe:.8f} (BẰNG 0 TUYỆT ĐỐI!)")

    print("\n2. Có Positional Encoding (X' = X + PE):")
    print(f"   Biểu diễn token 'love' trong 'I love NLP' : {np.round(out_a_with_pe[1], 6)}")
    print(f"   Biểu diễn token 'love' trong 'NLP love I' : {np.round(out_b_with_pe[1], 6)}")
    print(f"   Mean-pooled vector 'I love NLP' : {np.round(out_a_with_pe.mean(axis=0), 6)}")
    print(f"   Mean-pooled vector 'NLP love I' : {np.round(out_b_with_pe.mean(axis=0), 6)}")
    diff_with_pe = np.linalg.norm(out_a_with_pe.mean(axis=0) - out_b_with_pe.mean(axis=0))
    print(f"   -> Khoảng cách L2 giữa 2 vector toàn câu (có PE)   : {diff_with_pe:.6f} (> 0, PHÂN BIỆT RÕ RÀNG!)\n")

    print("=" * 75)
    print("MỤC 12: PERMUTATION EXPERIMENT ('dog bites man' vs 'man bites dog')")
    print("=" * 75)
    vocab_perm = {"dog": 0, "bites": 1, "man": 2}
    embed_perm = np.array([
        [1.2, -0.4, 0.7, 0.3],   # "dog"
        [-0.5, 1.1, 0.2, -0.6],  # "bites"
        [0.3, 0.6, -0.9, 1.0]    # "man"
    ])
    s1_tokens = ["dog", "bites", "man"]
    s2_tokens = ["man", "bites", "dog"]

    X_s1 = np.array([embed_perm[vocab_perm[w]] for w in s1_tokens])
    X_s2 = np.array([embed_perm[vocab_perm[w]] for w in s2_tokens])

    # Experiment A: Self-Attention KHÔNG Positional Encoding
    Z_s1_A, A_s1_A = simple_self_attention_numpy(X_s1, W_Q, W_K, W_V)
    Z_s2_A, A_s2_A = simple_self_attention_numpy(X_s2, W_Q, W_K, W_V)
    pool_s1_A = Z_s1_A.mean(axis=0)
    pool_s2_A = Z_s2_A.mean(axis=0)
    l2_exp_A = np.linalg.norm(pool_s1_A - pool_s2_A)
    # Kiểm tra tính chất hoán vị tương đương (Permutation Equivariance):
    # Hàng 0 của S1 ("dog") giống hệt hàng 2 của S2 ("dog")
    dog_diff_A = np.linalg.norm(Z_s1_A[0] - Z_s2_A[2])

    # Experiment B: Self-Attention + Positional Encoding
    Z_s1_B, A_s1_B = simple_self_attention_numpy(X_s1 + PE_3, W_Q, W_K, W_V)
    Z_s2_B, A_s2_B = simple_self_attention_numpy(X_s2 + PE_3, W_Q, W_K, W_V)
    pool_s1_B = Z_s1_B.mean(axis=0)
    pool_s2_B = Z_s2_B.mean(axis=0)
    l2_exp_B = np.linalg.norm(pool_s1_B - pool_s2_B)
    dog_diff_B = np.linalg.norm(Z_s1_B[0] - Z_s2_B[2])

    print("[Experiment A — Self-Attention KHÔNG Positional Encoding]")
    print(f"  S1 ('dog bites man') output Z_s1:\n{np.round(Z_s1_A, 4)}")
    print(f"  S2 ('man bites dog') output Z_s2:\n{np.round(Z_s2_A, 4)}")
    print(f"  Sai khác vector của từ 'dog' khi làm chủ ngữ (S1[0]) vs tân ngữ (S2[2]): {dog_diff_A:.8f}")
    print(f"  Mean-pooled S1: {np.round(pool_s1_A, 6)}")
    print(f"  Mean-pooled S2: {np.round(pool_s2_A, 6)}")
    print(f"  Khoảng cách L2 giữa S1 và S2 sau Mean Pooling: {l2_exp_A:.8f}")

    print("\n[Experiment B — Self-Attention CÓ Positional Encoding]")
    print(f"  S1 ('dog bites man') output Z_s1:\n{np.round(Z_s1_B, 4)}")
    print(f"  S2 ('man bites dog') output Z_s2:\n{np.round(Z_s2_B, 4)}")
    print(f"  Sai khác vector của từ 'dog' khi làm chủ ngữ (S1[0]) vs tân ngữ (S2[2]): {dog_diff_B:.6f}")
    print(f"  Mean-pooled S1: {np.round(pool_s1_B, 6)}")
    print(f"  Mean-pooled S2: {np.round(pool_s2_B, 6)}")
    print(f"  Khoảng cách L2 giữa S1 và S2 sau Mean Pooling: {l2_exp_B:.6f}")

    assert np.isclose(l2_exp_A, 0.0, atol=1e-7), "Experiment A không PE phải có pooled representation trùng nhau!"
    assert l2_exp_B > 0.1, "Experiment B có PE phải tạo ra representation khác biệt!"
    print("-> Kết luận thực nghiệm: Không có PE, Self-Attention có tính bất biến hoán vị sau pooling")
    print("   (Permutation Invariant), hoàn toàn mù thứ tự từ. Khi cộng PE, đối xứng hoán vị bị phá vỡ!\n")

    return {
        'Z_s1_A': Z_s1_A, 'Z_s2_A': Z_s2_A,
        'Z_s1_B': Z_s1_B, 'Z_s2_B': Z_s2_B,
        'pool_s1_A': pool_s1_A, 'pool_s2_A': pool_s2_A,
        'pool_s1_B': pool_s1_B, 'pool_s2_B': pool_s2_B,
        'l2_exp_A': float(l2_exp_A), 'l2_exp_B': float(l2_exp_B),
        'dog_diff_A': float(dog_diff_A), 'dog_diff_B': float(dog_diff_B)
    }


def save_positional_encoding_plot(save_path: str = "positional_encoding.png"):
    """
    Vẽ biểu đồ trực quan hóa Sinusoidal Positional Encoding và kết quả Permutation Experiment.
    Lưu vào file positional_encoding.png.
    """
    pe_matrix = compute_positional_encoding_numpy(max_len=64, d_model=128)
    perm_res = run_experiment_1_and_permutation()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1.5, 1.2, 1.1]})

    # Panel 1: Heatmap của ma trận PE (64 vị trí x 128 chiều)
    im = axes[0].imshow(pe_matrix, cmap='RdBu', aspect='auto', vmin=-1.0, vmax=1.0)
    axes[0].set_title("Sinusoidal Positional Encoding\n(max_len = 64, d_model = 128)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Embedding Dimension Index (2i, 2i+1)")
    axes[0].set_ylabel("Token Position (pos)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="PE Value [-1, 1]")

    # Panel 2: Các sóng sin/cos ở các tần số chiều khác nhau
    pos_range = np.arange(0, 64)
    for dim_idx, label in [(0, "dim 0 (sin, freq cao)"), (1, "dim 1 (cos, freq cao)"),
                           (16, "dim 16 (sin, freq trung)"), (32, "dim 32 (sin, freq thấp)")]:
        axes[1].plot(pos_range, pe_matrix[:, dim_idx], label=label, linewidth=1.8)
    axes[1].set_title("Sóng Sin/Cos theo Vị trí (pos)\nở các chiều tần số khác nhau", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Position (pos)")
    axes[1].set_ylabel("PE(pos, dim)")
    axes[1].legend(loc='upper right', fontsize=8.5)
    axes[1].grid(True, linestyle='--', alpha=0.4)

    # Panel 3: Minh chứng Permutation Experiment ("dog bites man" vs "man bites dog")
    categories = ["Pooled Sentence\nL2 Distance", "Token 'dog'\nL2 Distance"]
    no_pe_vals = [perm_res['l2_exp_A'], perm_res['dog_diff_A']]
    with_pe_vals = [perm_res['l2_exp_B'], perm_res['dog_diff_B']]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = axes[2].bar(x - width / 2, no_pe_vals, width, label="Exp A: No PE", color="#95a5a6", edgecolor="black")
    bars2 = axes[2].bar(x + width / 2, with_pe_vals, width, label="Exp B: With PE", color="#2980b9", edgecolor="black")

    axes[2].set_title("Permutation Experiment:\n'dog bites man' vs 'man bites dog'", fontsize=11, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories, fontsize=9.5)
    axes[2].set_ylabel("L2 Representation Distance")
    axes[2].set_ylim(0, max(with_pe_vals) * 1.25)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', linestyle='--', alpha=0.4)

    for b in bars1:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.4f}", ha='center', va='bottom', fontsize=8.5)
    for b in bars2:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.4f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu hình ảnh trực quan hóa: {save_path}")


if __name__ == "__main__":
    run_section_10_calculation()
    save_positional_encoding_plot("positional_encoding.png")

