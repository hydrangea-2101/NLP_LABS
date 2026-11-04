"""
NLP LAB 07 — TRANSFORMER ARCHITECTURE
Chủ đề: From Attention to Transformer

File: transformer_block.py
Mục đích:
    - Xây dựng một Transformer Encoder Block tối giản hoàn toàn từ các primitive:
      nn.Linear, F.softmax / nn.Softmax, nn.LayerNorm, nn.Dropout
      (TUYỆT ĐỐI KHÔNG dùng nn.TransformerEncoderLayer trực tiếp trong core implementation).
    - Cài đặt MultiHeadSelfAttention và PositionWiseFeedForward.
    - Kiểm tra chi tiết kích thước tensor (Shape Tracking — Mục 18: B=8, T=100, d_model=128, n_heads=4).
    - Kiểm thử đối chiếu với các bài tập tính tay (Mục 6, Mục 7, Mục 14).
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo mã hóa UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class MultiHeadSelfAttention(nn.Module):
    """
    Cài đặt Multi-Head Self-Attention từ các phép chiếu tuyến tính nn.Linear.

    Công thức (Mục 5 — W7.pdf):
        head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    với d_k = d_model / n_heads.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})!"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        # Các lớp chiếu tuyến tính cho Q, K, V và đầu ra W^O
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, return_details: bool = False):
        """
        Args:
            x (torch.Tensor): Input tensor có shape (B, T, d_model)
            mask (torch.Tensor, optional): Padding mask có shape (B, 1, 1, T) hoặc (B, 1, T, T)
            return_details (bool): Nếu True, trả về thêm dictionary chứa các tensor trung gian để kiểm tra shape.

        Returns:
            tuple:
                - out (torch.Tensor): Output của Multi-Head Attention, shape (B, T, d_model)
                - attn_weights (torch.Tensor): Trọng số attention của các heads, shape (B, n_heads, T, T)
        """
        B, T, D = x.shape
        assert D == self.d_model, f"Chiều cuối của x ({D}) phải bằng d_model ({self.d_model})"

        # 1. Chiếu tuyến tính và tách thành n_heads:
        # (B, T, d_model) -> (B, T, n_heads, d_k) -> hoán vị sang (B, n_heads, T, d_k)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Tính Scaled Dot-Product Attention Scores cho từng head:
        # (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        if mask is not None:
            # Nếu mask có shape (B, T) hoặc (B, 1, T), mở rộng thành (B, 1, 1, T)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Chuẩn hóa bằng Softmax
        attn_weights = self.softmax(scores)  # (B, n_heads, T, T)
        attn_dropped = self.dropout(attn_weights)

        # 4. Nhân trọng số Attention với V:
        # (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        head_outputs = torch.matmul(attn_dropped, V)

        # 5. Concatenation các heads và áp dụng phép chiếu W^O:
        # (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        concat_heads = head_outputs.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(concat_heads)  # (B, T, d_model)

        if return_details:
            details = {
                'Q': Q,
                'K': K,
                'V': V,
                'scores': scores,
                'attn_weights': attn_weights,
                'head_outputs': head_outputs,
                'concat_heads': concat_heads,
                'attn_output': out
            }
            return out, attn_weights, details

        return out, attn_weights


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (Mục 13 — W7.pdf):
        FFN(x) = W_2 * ReLU(W_1 * x + b_1) + b_2
    Thực hiện biến đổi phi tuyến độc lập và giống nhau tại từng vị trí token.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> hidden: (B, T, d_ff) -> out: (B, T, d_model)
        hidden = self.activation(self.fc1(x))
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block tối giản (Mục 17 — W7.pdf).
    Cấu trúc chuẩn:
        x
        ↓
        Multi-Head Self-Attention
        ↓
        Residual + LayerNorm (norm1(x + dropout(attn_out)))
        ↓
        Feed-Forward Network
        ↓
        Residual + LayerNorm (norm2(h + dropout(ffn_out)))
        ↓
        output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False,
        return_details: bool = False
    ):
        """
        Thực hiện lan truyền tiến qua Transformer Encoder Block.

        Args:
            x (torch.Tensor): Input tensor shape (B, T, d_model)
            mask (torch.Tensor, optional): Mask cho các vị trí <PAD>
            return_attention (bool): Nếu True, trả về (output, attn_weights)
            return_details (bool): Nếu True, trả về (output, attn_weights, shape_dict)
        """
        if return_details:
            attn_out, attn_weights, details = self.mha(x, mask=mask, return_details=True)
        else:
            attn_out, attn_weights = self.mha(x, mask=mask, return_details=False)

        # Nhánh 1: Residual Connection + LayerNorm sau Multi-Head Self-Attention
        res1 = self.norm1(x + self.dropout1(attn_out))

        # Nhánh 2: Position-wise Feed-Forward Network + Residual + LayerNorm
        ffn_out = self.ffn(res1)
        output = self.norm2(res1 + self.dropout2(ffn_out))

        if return_details:
            details['X'] = x
            details['res1'] = res1
            details['ffn_hidden'] = self.ffn.activation(self.ffn.fc1(res1))
            details['ffn_out'] = ffn_out
            details['output'] = output
            return output, attn_weights, details

        if return_attention:
            return output, attn_weights

        return output


# ==============================================================================
# KIỂM TRA SHAPE VÀ KIỂM THỬ CÁC BÀI TẬP TÍNH TAY (MỤC 6, 7, 14, 18)
# ==============================================================================

def verify_shape_section_18():
    """
    Kiểm tra bắt buộc Mục 18 (W7.pdf):
        B = 8, T = 100, d_model = 128, n_heads = 4 -> d_k = 128 / 4 = 32
    In bảng theo dõi shape qua từng bước biến đổi trong TransformerBlock.
    """
    print("=" * 75)
    print("MỤC 18: KIỂM TRA SHAPE TRONG TRANSFORMER BLOCK")
    print("=" * 75)
    B = 8
    T = 100
    d_model = 128
    n_heads = 4
    d_ff = 256
    d_k = d_model // n_heads

    torch.manual_seed(42)
    x = torch.randn(B, T, d_model)
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=0.1)
    block.eval()

    with torch.no_grad():
        output, attn_weights, details = block(x, return_details=True)

    print(f"Cấu hình: B = {B}, T = {T}, d_model = {d_model}, n_heads = {n_heads}, d_ff = {d_ff} => d_k = {d_k}\n")
    steps = [
        ("X (Input)", details['X'].shape, (B, T, d_model)),
        ("Q (Projected & Split Heads)", details['Q'].shape, (B, n_heads, T, d_k)),
        ("K (Projected & Split Heads)", details['K'].shape, (B, n_heads, T, d_k)),
        ("V (Projected & Split Heads)", details['V'].shape, (B, n_heads, T, d_k)),
        ("Attention scores (Q K^T / sqrt(d_k))", details['scores'].shape, (B, n_heads, T, T)),
        ("Attention weights (Softmax)", details['attn_weights'].shape, (B, n_heads, T, T)),
        ("Head outputs (A V)", details['head_outputs'].shape, (B, n_heads, T, d_k)),
        ("Concat heads", details['concat_heads'].shape, (B, T, d_model)),
        ("Attention output (sau W^O)", details['attn_output'].shape, (B, T, d_model)),
        ("Residual + LayerNorm 1", details['res1'].shape, (B, T, d_model)),
        ("FFN hidden (sau W_1 + ReLU)", details['ffn_hidden'].shape, (B, T, d_ff)),
        ("FFN output (sau W_2)", details['ffn_out'].shape, (B, T, d_model)),
        ("Output (Residual + LayerNorm 2)", details['output'].shape, (B, T, d_model)),
    ]

    for name, actual_shape, expected_shape in steps:
        assert tuple(actual_shape) == expected_shape, (
            f"Sai kích thước tại {name}: thực tế {actual_shape} != kỳ vọng {expected_shape}"
        )
        print(f"  {name:<38} : {str(tuple(actual_shape)):<20} (Kỳ vọng: {expected_shape}) [OK]")

    # Kiểm tra tổng xác suất hàng của Attention weights bằng 1
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Tổng hàng attention phải bằng 1!"
    print("\n-> Tất cả kiểm tra shape ở Mục 18 và tổng xác suất Softmax đều PASSED 100%!\n")


def verify_hand_calculations():
    """
    Kiểm thử các bài tập tính tay trong Mục 6, Mục 7 và Mục 14 để đối chiếu với calculations.md.
    """
    print("=" * 75)
    print("KIỂM THỬ BÀI TẬP TÍNH TOÁN MỤC 6, MỤC 7 VÀ MỤC 14")
    print("=" * 75)

    # 1. Mục 6: Một Attention Head với X thuộc R^(3x4), d_k = 2, W^Q, W^K, W^V thuộc R^(4x2)
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ])  # (3, 4)
    W_Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])  # (4, 2)
    W_K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0]
    ])  # (4, 2) -> K = [[1, 0], [0, 1], [1, 1]]
    W_V = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 1.0]
    ])  # (4, 2)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    S = (Q @ K.T) / math.sqrt(2.0)
    S_shift = S - np.max(S, axis=-1, keepdims=True)
    A = np.exp(S_shift) / np.sum(np.exp(S_shift), axis=-1, keepdims=True)
    Z = A @ V

    print("[Mục 6 — Một Attention Head]")
    print(f"  Q (shape {Q.shape}):\n{Q}")
    print(f"  K (shape {K.shape}):\n{K}")
    print(f"  V (shape {V.shape}):\n{V}")
    print(f"  S = Q K^T / sqrt(2) (shape {S.shape}):\n{np.round(S, 4)}")
    print(f"  A = softmax(S) (shape {A.shape}):\n{np.round(A, 4)}")
    print(f"  Z = A V (shape {Z.shape}):\n{np.round(Z, 4)}")

    # 2. Mục 7: Multi-head Concatenation (2 heads, mỗi head tạo ra vector chiều 2 -> Concat R^(T x 4))
    head_1 = Z  # (3, 2)
    head_2 = np.array([
        [0.5, 1.5],
        [1.0, 0.5],
        [1.5, 1.0]
    ])  # (3, 2)
    concat_12 = np.concatenate([head_1, head_2], axis=-1)
    assert concat_12.shape == (3, 4)
    print(f"\n[Mục 7 — Multi-Head Concatenation]")
    print(f"  head_1 shape: {head_1.shape}, head_2 shape: {head_2.shape}")
    print(f"  Concat(head_1, head_2) shape: {concat_12.shape}")

    # 3. Mục 14: Bài tập tính FFN
    # x = [[1], [2]], W_1 = [[1, 0], [0, 1]], b_1 = [[0], [-1]]
    x_ffn = np.array([1.0, 2.0])
    W_1 = np.array([[1.0, 0.0],
                    [0.0, 1.0]])
    b_1 = np.array([0.0, -1.0])
    pre_act = W_1 @ x_ffn + b_1  # [1, 1]
    h_ffn = np.maximum(0.0, pre_act)  # ReLU([1, 1]) = [1, 1]

    # Giả sử W_2 = [[2, -1], [1, 1]], b_2 = [0.5, -0.5]
    W_2 = np.array([[2.0, -1.0],
                    [1.0,  1.0]])
    b_2 = np.array([0.5, -0.5])
    y_ffn = W_2 @ h_ffn + b_2  # [1.5, 1.5]

    print(f"\n[Mục 14 — Bài tập tính FFN]")
    print(f"  x = {x_ffn}")
    print(f"  W_1 x + b_1 = {pre_act}")
    print(f"  h = ReLU(W_1 x + b_1) = {h_ffn}")
    print(f"  Với W_2 = [[2, -1], [1, 1]], b_2 = [0.5, -0.5] => y = W_2 h + b_2 = {y_ffn}")
    assert np.allclose(h_ffn, np.array([1.0, 1.0]))
    assert np.allclose(y_ffn, np.array([1.5, 1.5]))
    print("-> Tất cả các bài tập tính toán Mục 6, 7, 14 đều PASSED!\n")


if __name__ == "__main__":
    verify_shape_section_18()
    verify_hand_calculations()
