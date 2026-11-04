# LAB 07: BÁO CÁO PHẢN TƯ / ĐÚC KẾT HỌC TẬP (REFLECTION)

### 1. Tại sao Transformer cần Positional Encoding?
- Cơ chế Self-Attention tính tương quan giữa các token qua tích vô hướng $q_i^T k_j$ và tổng hợp ngữ cảnh bằng tổ hợp tuyến tính $\sum_{j=1}^T \alpha_{ij} v_j$. Cả phép nhân ma trận này lẫn mạng Position-wise FFN đều hoàn toàn độc lập với chỉ số vị trí của token trong chuỗi.
- Nếu không cộng thêm Positional Encoding ($X' = X + PE$), Transformer sẽ có tính **bất biến hoán vị (Permutation Invariant)** sau bước Mean Pooling: hai câu có cùng từ vựng nhưng ngược nghĩa như *"dog bites man"* và *"man bites dog"* sẽ tạo ra cùng một vector đại diện giống hệt nhau (khoảng cách L2 bằng `0.0000` trong thực nghiệm).
- Sinusoidal Positional Encoding sử dụng cặp hàm lượng giác $\sin\left(\frac{pos}{10000^{2i/d}}\right)$ và $\cos\left(\frac{pos}{10000^{2i/d}}\right)$ ở nhiều tần số khác nhau để gán cho mỗi vị trí một vector định danh duy nhất, phá vỡ tính đối xứng hoán vị và giúp mô hình nắm bắt cả vị trí tuyệt đối lẫn khoảng cách tương đối giữa các từ.

### 2. Multi-Head Attention khác một Attention Head ở đâu và vai trò của $d_k$?
- **Một Attention Head (Single-Head):** Chỉ thực hiện một phép chiếu tuyến tính duy nhất, tạo ra một ma trận Attention $T \times T$ duy nhất cho toàn câu.
- **Multi-Head Attention:** Chia không gian biểu diễn $d_{\text{model}}$ thành $h$ không gian con độc lập có chiều:
  $$d_k = \frac{d_{\text{model}}}{n_{\text{heads}}}$$
  Ví dụ: khi $d_{\text{model}} = 128$ và có $n_{\text{heads}} = 8$ thì mỗi head có chiều $d_k = \frac{128}{8} = 16$.
- Mỗi head $\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$ học một kiểu quan hệ ngữ cảnh riêng biệt song song (ví dụ trong Experiment 3: Head 1 tập trung vào tính từ cảm xúc `"great"`, Head 2 chú ý vào cặp phủ định - chuyển ý `"not" -> "ending"`).
- Phép ghép nối $\text{Concat}(\text{head}_1, \dots, \text{head}_h) \in \mathbb{R}^{T \times d_{\text{model}}}$ bảo toàn nguyên vẹn thông tin từ mọi không gian con trước khi nhân với ma trận chiếu $W^O$ để trộn thông tin liên head (cross-head mixing). Việc tăng số heads khi cố định $d_{\text{model}}$ không làm tăng số chiều biểu diễn hay tổng số tham số (`2,060,801`), và cũng không đảm bảo accuracy tăng mãi (đạt đỉnh `79.80%` ở 2 heads, `79.20%` ở 4 heads và giảm nhẹ còn `78.30%` ở 8 heads khi $d_k$ thu nhỏ xuống 16).

### 3. Vai trò của Feed-Forward Network (FFN), Residual Connection và LayerNorm?
- **Position-wise Feed-Forward Network ($\text{FFN}(x) = W_2 \text{ReLU}(W_1 x + b_1) + b_2$):** Trong khi Self-Attention chịu trách nhiệm **trộn thông tin giữa các vị trí token** (token mixing), FFN thực hiện phép biến đổi **phi tuyến độc lập tại từng vị trí** (feature/channel mixing), mở rộng chiều từ $d_{\text{model}} \to d_{\text{ff}}$ rồi chiếu về $d_{\text{model}}$ để trích xuất đặc trưng bậc cao.
- **Residual Connection ($y = x + F(x)$):** Khác với việc chỉ dùng $F(x)$ (dễ làm biến dạng thông tin gốc và Positional Encoding qua nhiều phép chiếu), đường truyền tắt $x + F(x)$ giữ nguyên biểu diễn đầu vào và tạo một "đường cao tốc đạo hàm" ($\frac{\partial y}{\partial x} = I + \frac{\partial F(x)}{\partial x}$), ngăn chặn hiện tượng triệt tiêu gradient (vanishing gradient) và giúp tầng con chỉ cần học phần hiệu chỉnh dư $\Delta x = F(x)$.
- **Layer Normalization:** Chuẩn hóa vector đặc trưng của từng token độc lập theo chiều $d_{\text{model}}$ về kỳ vọng 0 và phương sai 1, giúp ổn định phân phối kích hoạt và tăng tốc độ hội tụ.

### 4. Tại sao Transformer xử lý song song hơn RNN nhưng vẫn bị giới hạn bởi $\mathcal{O}(T^2)$?
- **Tính song song:** RNN/LSTM phải tính tuần tự $h_1 \to h_2 \to \dots \to h_T$, trong khi Transformer tính toàn bộ $T$ vị trí cùng lúc bằng nhân ma trận. Ở `max_length = 128`, Transformer huấn luyện nhanh gần gấp đôi LSTM (`46.90s` so với `91.77s`) và đạt F1 cao hơn hẳn (`0.7899` so với `0.6790`).
- **Giới hạn bậc hai $\mathcal{O}(T^2)$:** Ma trận Attention của mỗi head có kích thước $T \times T$ (do nhân $Q \in \mathbb{R}^{T \times d_k}$ với $K^T \in \mathbb{R}^{d_k \times T}$). Khi độ dài chuỗi tăng gấp đôi từ $128 \to 256$, số phần tử trong ma trận Attention tăng gấp **4 lần** ($128^2 = 16.384 \to 256^2 = 65.536$), bộ nhớ Attention mỗi batch tăng từ `16 MB` lên `64 MB`, và thời gian huấn luyện tăng vọt từ `46.90s` lên `166.22s`. Ở chuỗi rất dài ($T = 1024$), Transformer thậm chí chậm hơn LSTM gấp 6 lần (`194.58 ms` so với `32.52 ms` mỗi forward pass). Do đó, Transformer không thể giải quyết mọi bài toán chỉ bằng cách tăng vô hạn `max_length`.

### 5. Vì sao không thể đồng nhất Attention Weight cao với "Explanation"?
- **HOÀN TOÀN KHÔNG ĐỒNG NHẤT (`Attention != explanation`).**
- Cần phân biệt rõ:
  - *Model Behavior:* Trọng số Attention chỉ là hệ số tổ hợp lồi dùng để trộn các vector Value $V$ bên trong nhánh Self-Attention. Sau đó, biểu diễn còn được cộng trực tiếp với đầu vào gốc qua **Residual Connection**, biến đổi phi tuyến qua **FFN**, và gộp qua **Mean Pooling** trước khi vào tầng phân loại.
  - *Human Interpretation:* Trong Error Analysis, ở **Sample #1**, từ tiêu cực `"worse"` có Attention weight cao nhất câu (`0.070`) nhưng mô hình vẫn đoán **Positive**; ngược lại ở **Sample #11**, từ tích cực `"nice"` có Attention weight cao nhất (`0.070`) nhưng mô hình vẫn đoán **Negative**. Do đó, Attention visualization chỉ hiển thị phân phối trọng số nội bộ, không phải là lời giải thích nhân quả cho dự đoán của mô hình.

---

- **Chuỗi giải quyết hạn chế qua từng bài Lab:**
  - **TF-IDF $\to$ Limitation:** Biểu diễn thưa (sparse), chiều cực lớn, mất thứ tự từ và không hiểu độ tương đồng ngữ nghĩa giữa các từ đồng nghĩa.
  - **Word2Vec $\to$ Limitation:** Vector đặc (dense) nhưng **tĩnh (static)** — mỗi từ chỉ có một vector cố định không đổi theo ngữ cảnh; khi lấy trung bình toàn câu sẽ mất sạch thứ tự từ.
  - **LSTM $\to$ Limitation:** Xử lý tuần tự qua từng bước thời gian $t$ nên không thể tính toán song song (training chậm) và gặp nút thắt cổ chai khi phải nén câu dài qua chuỗi trạng thái ẩn $h_t$.
  - **Attention $\to$ Limitation:** Tương tác trực tiếp mọi cặp token song song nhưng thiếu thông tin vị trí (mù thứ tự), chỉ có một không gian chiếu đơn lẻ và thiếu mạng biến đổi phi tuyến sâu cùng cơ chế ổn định gradient.
  - **Transformer $\to$ Remaining limitation:** Chi phí bộ nhớ và tính toán bậc hai $\mathcal{O}(T^2)$ theo chiều dài chuỗi; đồng thời khi **huấn luyện từ đầu (from scratch)** trên tập dữ liệu nhỏ (5.000 mẫu), mô hình rất nhanh bị **overfitting** (Validation Loss tăng ngược từ Epoch 3 trong `learning_curves.png`) và thiếu tri thức ngôn ngữ nền tảng.

---
