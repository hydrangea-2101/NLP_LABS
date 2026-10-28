"""
NLP LAB 06 — ATTENTION MECHANISM FROM SCRATCH (NUMPY IMPLEMENTATION)
Chủ đề: From Recurrent State to Context-Aware Representation

File: attention_numpy.py
Mục đích:
    - Cài đặt cơ chế Scaled Dot-Product Attention hoàn toàn bằng thư viện NumPy.
    - Đảm bảo tính ổn định số học (numerical stability) cho hàm Softmax.
    - Cung cấp các kiểm thử (unit tests) đối chiếu chính xác với các bài tập tính tay (Bài 1 -> 4, Scaling).
    - Giải thích chi tiết các bước toán học và shape của tensor bằng tiếng Việt.
"""

import sys
import numpy as np

# Đảm bảo mã hóa UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass



def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Hàm tính Softmax ổn định số học (Numerically Stable Softmax).
    
    Công thức:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        
    Giải thích toán học vì sao trừ max(x) không làm đổi kết quả:
        exp(x_i - c) / sum_j exp(x_j - c)
        = [exp(x_i) * exp(-c)] / [sum_j (exp(x_j) * exp(-c))]
        = exp(-c) * exp(x_i) / [exp(-c) * sum_j exp(x_j)]
        = exp(x_i) / sum_j exp(x_j)
    Nhờ trừ max(x), số mũ lớn nhất là exp(0) = 1, triệt tiêu hoàn toàn nguy cơ
    tràn số dương (overflow -> inf/nan) khi gặp giá trị score lớn (ví dụ: x = 1000).
    
    Args:
        x (np.ndarray): Mảng đầu vào (1D, 2D hoặc 3D).
        axis (int): Trục thực hiện chuẩn hóa Softmax (mặc định là trục cuối cùng -1).
        
    Returns:
        np.ndarray: Phân phối xác suất Softmax có cùng kích thước với x, tổng trên axis bằng 1.
    """
    # Trừ giá trị lớn nhất theo trục axis (keepdims=True để broadcasting đúng shape)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(x - x_max)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / sum_exp


def compute_scores(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Tính Attention Scores bằng phép nhân vô hướng (Dot Product).
    
    Score đo mức độ tương thích giữa Query và Key.
    Với Q có shape (..., T_q, d_k) và K có shape (..., T_k, d_k):
        Score = Q @ K^T có shape (..., T_q, T_k)
        
    Args:
        Q (np.ndarray): Ma trận Query, kích thước (..., T_q, d_k)
        K (np.ndarray): Ma trận Key, kích thước (..., T_k, d_k)
        
    Returns:
        np.ndarray: Ma trận Attention Scores thô, kích thước (..., T_q, T_k)
    """
    assert Q.shape[-1] == K.shape[-1], (
        f"Lỗi kích thước: Chiều vector của Q ({Q.shape[-1]}) "
        f"phải khớp với chiều vector của K ({K.shape[-1]})"
    )
    # Hoán vị 2 chiều cuối của K để thực hiện phép nhân ma trận Q @ K^T
    return np.matmul(Q, np.swapaxes(K, -1, -2))


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray = None,
    temperature: float = 1.0
):
    """
    Tính toán Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax((Q K^T) / (sqrt(d_k) * T)) V
        
    Quy trình tính toán:
        1. Tính ma trận điểm số Dot-product: S_raw = Q @ K^T
        2. Chia cho hệ số tỉ lệ sqrt(d_k) và temperature: S_scaled = S_raw / (sqrt(d_k) * T)
        3. Áp dụng Mask (nếu có): cộng -1e9 vào các vị trí bị che (ví dụ padding tokens)
        4. Chuẩn hóa xác suất qua hàm stable_softmax: A = softmax(S_scaled)
        5. Tính context vector / output: C = A @ V
        
    Args:
        Q (np.ndarray): Query tensor, shape (..., T_q, d_k)
        K (np.ndarray): Key tensor, shape (..., T_k, d_k)
        V (np.ndarray): Value tensor, shape (..., T_k, d_v)
        mask (np.ndarray, optional): Mask tensor (1: giữ, 0: che), broadcastable với shape (..., T_q, T_k)
        temperature (float): Hệ số nhiệt độ T > 0 để điều chỉnh độ nhọn của phân phối Softmax.
        
    Returns:
        tuple (scores, attention_weights, context):
            - scores: Điểm số đã scale (S_scaled), shape (..., T_q, T_k)
            - attention_weights: Trọng số phân phối (A), shape (..., T_q, T_k)
            - context: Biểu diễn ngữ cảnh đầu ra (C), shape (..., T_q, d_v)
    """
    # 1. Kiểm tra tính hợp lệ của kích thước tensor
    d_k = K.shape[-1]
    assert Q.shape[-1] == d_k, f"Chiều của Q ({Q.shape[-1]}) khác chiều d_k của K ({d_k})"
    assert K.shape[-2] == V.shape[-2], f"Số lượng Key ({K.shape[-2]}) phải bằng số lượng Value ({V.shape[-2]})"
    assert temperature > 0, "Temperature T phải là số thực dương > 0"
    
    # 2. Tính raw dot product và scale với sqrt(d_k) cùng temperature
    raw_scores = compute_scores(Q, K)
    scale_factor = np.sqrt(d_k) * temperature
    scaled_scores = raw_scores / scale_factor
    
    # 3. Xử lý Mask (nếu có mask 0 cho các vị trí bị loại bỏ)
    if mask is not None:
        # Gán giá trị âm cực lớn (-1e9) tại vị trí mask == 0 để softmax gán trọng số xấp xỉ 0
        scaled_scores = np.where(mask == 0, -1e9, scaled_scores)
        
    # 4. Tính Attention Weights bằng stable softmax dọc theo chiều Key (axis=-1)
    attention_weights = stable_softmax(scaled_scores, axis=-1)
    
    # 5. Tính Context Vector bằng tổ hợp tuyến tính các Value: C = A @ V
    context = np.matmul(attention_weights, V)
    
    return scaled_scores, attention_weights, context


# ==============================================================================
# PHẦN KIỂM THỬ ĐỐI CHIẾU VỚI BÀI TẬP TÍNH TOÁN TRONG W6.PDF & CACULATIONS_LAB06.PDF
# ==============================================================================

def run_tests():
    print("=" * 70)
    print("KIỂM THỬ 1: TÍNH ỔN ĐỊNH SỐ HỌC CỦA SOFTMAX (NUMERICAL STABILITY)")
    print("=" * 70)
    x_large = np.array([1000.0, 1001.0, 1002.0])
    
    # Thử gọi np.exp trực tiếp mà không trừ max (sẽ bị Overflow)
    with np.errstate(over='ignore'):
        naive_exp = np.exp(x_large)
    print(f"Vector x lớn: {x_large}")
    print(f"Tính naive np.exp(x): {naive_exp} (Bị tràn số inf!)")
    
    # Tính bằng stable_softmax
    weights_stable = stable_softmax(x_large)
    print(f"Kết quả stable_softmax(x): {weights_stable}")
    print(f"Tổng xác suất: {np.sum(weights_stable):.6f}")
    assert np.allclose(np.sum(weights_stable), 1.0), "Tổng xác suất softmax phải bằng 1!"
    print("-> Test Numerical Stability: PASSED!\n")

    print("=" * 70)
    print("KIỂM THỬ 2: BÀI TẬP 1 — ATTENTION SCORES")
    print("=" * 70)
    # q = [1, 2]^T, K = [[1, 0], [0, 1], [1, 1]]
    q1 = np.array([[1.0, 2.0]])         # Shape (1, 2)
    K1 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])         # Shape (3, 2)
    s1 = compute_scores(q1, K1)[0]       # Shape (3,)
    expected_s1 = np.array([1.0, 2.0, 3.0])
    print(f"Query q: {q1[0]}")
    print(f"Keys K:\n{K1}")
    print(f"Score vector s = q @ K^T: {s1}")
    print(f"Đáp án lý thuyết: {expected_s1}")
    assert np.allclose(s1, expected_s1), "Score vector không khớp đáp án lý thuyết!"
    print("-> Test Bài 1: PASSED!\n")

    print("=" * 70)
    print("KIỂM THỬ 3: BÀI TẬP 2 — SOFTMAX")
    print("=" * 70)
    # s = [1, 2, 3] -> alpha = softmax(s)
    s2 = np.array([1.0, 2.0, 3.0])
    alpha2 = stable_softmax(s2)
    expected_alpha2 = np.array([
        np.exp(1) / (np.exp(1) + np.exp(2) + np.exp(3)),
        np.exp(2) / (np.exp(1) + np.exp(2) + np.exp(3)),
        np.exp(3) / (np.exp(1) + np.exp(2) + np.exp(3))
    ])
    print(f"Score vector s: {s2}")
    print(f"Softmax weights alpha: {alpha2}")
    print(f"Đáp án xấp xỉ trong caculations_lab06.pdf: [0.0900, 0.2447, 0.6652]")
    print(f"Tổng alpha_i: {np.sum(alpha2):.6f}")
    assert np.allclose(alpha2, expected_alpha2), "Softmax weights không khớp!"
    assert np.allclose(np.sum(alpha2), 1.0), "Tổng weights phải bằng 1!"
    print("-> Test Bài 2: PASSED!\n")

    print("=" * 70)
    print("KIỂM THỬ 4: BÀI TẬP 3 — CONTEXT VECTOR")
    print("=" * 70)
    # alpha = [0.1, 0.2, 0.7], v1=[1, 0], v2=[0, 1], v3=[2, 2]
    alpha3 = np.array([[0.1, 0.2, 0.7]])  # Shape (1, 3)
    V3 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [2.0, 2.0]])           # Shape (3, 2)
    c3 = np.matmul(alpha3, V3)[0]         # Shape (2,)
    expected_c3 = 0.1 * V3[0] + 0.2 * V3[1] + 0.7 * V3[2]
    print(f"Attention weights alpha: {alpha3[0]}")
    print(f"Values V:\n{V3}")
    print(f"Context vector c: {c3}")
    print(f"Đáp số bài viết tay trong caculations_lab06.pdf: [1.5, 1.6]")
    assert np.allclose(c3, expected_c3), "Context vector không khớp!"
    assert np.allclose(c3, np.array([1.5, 1.6])), "Đáp số tính toán không phải [1.5, 1.6]!"
    print("-> Test Bài 3: PASSED!\n")

    print("=" * 70)
    print("KIỂM THỬ 5: BÀI TẬP 4 — TOÀN BỘ ATTENTION COMPUTATION")
    print("=" * 70)
    # q = [1, 0], k1=[1, 0], k2=[0, 1], k3=[1, 1], v1=[1, 0], v2=[0, 2], v3=[2, 1]
    q4 = np.array([[1.0, 0.0]])           # Shape (1, 2)
    K4 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])           # Shape (3, 2)
    V4 = np.array([[1.0, 0.0],
                   [0.0, 2.0],
                   [2.0, 1.0]])           # Shape (3, 2)
    
    # Ở Bài 4, đề bài dùng dot-product thuần túy (không scale với sqrt(d_k))
    scores4_raw = compute_scores(q4, K4)
    weights4 = stable_softmax(scores4_raw)
    context4 = np.matmul(weights4, V4)[0]
    
    print(f"Query q: {q4[0]}")
    print(f"Scores s = q @ K^T: {scores4_raw[0]} (Kỳ vọng: [1, 0, 1])")
    print(f"Attention weights alpha: {weights4[0]} (Kỳ vọng: [0.4223, 0.1554, 0.4223])")
    print(f"Context vector c: {context4} (Kỳ vọng: [1.2669, 0.7331])")
    assert np.allclose(scores4_raw[0], np.array([1.0, 0.0, 1.0]))
    assert np.allclose(weights4[0], np.array([0.422319, 0.155362, 0.422319]), atol=1e-4)
    assert np.allclose(context4, np.array([1.266957, 0.733043]), atol=1e-4)
    print("-> Test Bài 4: PASSED!\n")

    print("=" * 70)
    print("KIỂM THỬ 6: BÀI TẬP SCALING & TEMPERATURE (MỤC 14 & 15)")
    print("=" * 70)
    # QK^T = [4, 2, 0], d_k = 4 -> sqrt(d_k) = 2
    raw_qk = np.array([4.0, 2.0, 0.0])
    weights_no_scale = stable_softmax(raw_qk)
    weights_scaled = stable_softmax(raw_qk / np.sqrt(4.0))
    
    print(f"Raw QK^T: {raw_qk}")
    print(f"Softmax KHÔNG scaling: {weights_no_scale}")
    print(f"  -> Max weight = {np.max(weights_no_scale):.4f}")
    print(f"Softmax CÓ scaling (chia sqrt(4)=2): {weights_scaled}")
    print(f"  -> Max weight = {np.max(weights_scaled):.4f}")
    print("Nhận xét: Scaling làm giảm độ phân hóa cực đoan, giúp phân phối mượt hơn.")
    
    # Kiểm tra Temperature với score vector s = [4, 2, 0]
    print("\nThử nghiệm Temperature:")
    for T in [0.5, 1.0, 2.0]:
        w = stable_softmax(raw_qk / T)
        entropy = -np.sum(w * np.log(w + 1e-12))
        print(f"  T = {T:4.1f} | Max weight: {np.max(w):.4f} | Entropy: {entropy:.4f} | Weights: {np.round(w, 4)}")
    print("-> Test Scaling & Temperature: PASSED!\n")

    print("=" * 70)
    print("KIỂM THỬ 7: KIỂM TRA BATCH SHAPE & TỔNG TRỌNG SỐ CHO TENSOR BẬC 3")
    print("=" * 70)
    # Shape batch: B = 2, T_q = 4, T_k = 6, d_k = 8, d_v = 16
    B, T_q, T_k, d_k, d_v = 2, 4, 6, 8, 16
    np.random.seed(42)
    Q_batch = np.random.randn(B, T_q, d_k)
    K_batch = np.random.randn(B, T_k, d_k)
    V_batch = np.random.randn(B, T_k, d_v)
    
    scores, attn, out = scaled_dot_product_attention(Q_batch, K_batch, V_batch)
    print(f"Kích thước Q: {Q_batch.shape}")
    print(f"Kích thước K: {K_batch.shape}")
    print(f"Kích thước V: {V_batch.shape}")
    print(f"Kích thước Scores: {scores.shape} (Kỳ vọng: ({B}, {T_q}, {T_k}))")
    print(f"Kích thước Attention Weights: {attn.shape} (Kỳ vọng: ({B}, {T_q}, {T_k}))")
    print(f"Kích thước Output Context: {out.shape} (Kỳ vọng: ({B}, {T_q}, {d_v}))")
    
    assert scores.shape == (B, T_q, T_k)
    assert attn.shape == (B, T_q, T_k)
    assert out.shape == (B, T_q, d_v)
    
    # Kiểm tra tổng trọng số mỗi hàng dọc trục Key bằng 1
    row_sums = np.sum(attn, axis=-1)
    assert np.allclose(row_sums, 1.0), "Tổng hàng của attention weights phải bằng 1!"
    print(f"Kiểm tra tổng hàng (min={np.min(row_sums):.6f}, max={np.max(row_sums):.6f}) xấp xỉ 1: CHÍNH XÁC!")
    print("-> Test Batch Shape: PASSED!\n")
    print("=" * 70)
    print("TẤT CẢ UNIT TESTS ĐỀU ĐẠT THÀNH CÔNG 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
