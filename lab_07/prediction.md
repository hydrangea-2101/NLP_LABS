# LAB 07: PREDICTION TRƯỚC EXPERIMENT

---

### Prediction 1: Khả năng xử lý song song của Transformer so với RNN
- **Câu hỏi:** Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?
- **Dự đoán (Prediction):** **CÓ.** Transformer có thể xử lý toàn bộ các token trong một sequence song song cùng lúc trong quá trình huấn luyện, vượt trội hơn hẳn so với tính chất tuần tự qua từng bước thời gian của RNN/LSTM.
- **Lý do (Reason):**  
  1. Trong mạng hồi quy RNN/LSTM, trạng thái ẩn tại bước thời gian $t$ phụ thuộc trực tiếp vào trạng thái ẩn trước đó:
     $$h_t = f(x_t, h_{t-1})$$
     Sự phụ thuộc tuần tự này buộc mô hình phải tính lần lượt từ $t = 1 \to T$, không thể song song hóa theo chiều dài chuỗi.
  2. Trong Transformer Encoder, toàn bộ các phép tính từ chiếu tuyến tính $Q = X W^Q, K = X W^K, V = X W^V$, nhân ma trận $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$, chuẩn hóa LayerNorm cho đến mạng Position-wise FFN đều được thực hiện đồng thời cho mọi vị trí token bằng các phép nhân tensor song song.
- **Mức độ tự tin (Confidence):** 100% (Đặc tính kiến trúc cốt lõi của Transformer).

---

### Prediction 2: Vai trò của Positional Encoding đối với thứ tự token
- **Câu hỏi:** Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?
- **Dự đoán (Prediction):** **KHÔNG.** Nếu loại bỏ hoàn toàn Positional Encoding, một khối Transformer Encoder thuần túy sẽ không còn bất kỳ khái niệm nào về thứ tự vị trí của các token.
- **Lý do (Reason):**  
  - Phép tính tích vô hướng $q_i^T k_j$ và tổ hợp tuyến tính $\sum_{j=1}^T \alpha_{ij} v_j$ trong Self-Attention chỉ phụ thuộc vào nội dung vector của các từ mà hoàn toàn độc lập với chỉ số vị trí $i, j$.
  - Tầng Position-wise FFN và LayerNorm cũng chỉ áp dụng cùng một phép biến đổi độc lập lên từng vector token. Do đó, nếu không cộng thêm Positional Encoding ($X' = X + PE$), mô hình chỉ xem câu đầu vào như một tập hợp từ không có thứ tự (bag/set of tokens).
- **Mức độ tự tin (Confidence):** 100% (Bảo đảm bằng công thức toán học).

---

### Prediction 3: Hai câu cùng tập token nhưng khác thứ tự khi không có Positional Information
- **Câu hỏi:** Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?
- **Dự đoán (Prediction):** **KHÔNG.** Sau bước gộp chuỗi (Mean/Sum Pooling), biểu diễn vector của hai câu có cùng tập từ vựng nhưng khác thứ tự sẽ trùng khít hoàn toàn 100%; ở mức từng token, các hàng của ma trận đầu ra chỉ bị đổi chỗ tương ứng với phép hoán vị đầu vào.
- **Lý do (Reason):**  
  - Gọi $P \in \mathbb{R}^{T \times T}$ là ma trận hoán vị đổi thứ tự các từ trong câu ($X_{\pi} = P X$). Khi không có Positional Encoding:
    $$\text{Attention}(P X) = \text{softmax}\left(\frac{(P Q)(P K)^T}{\sqrt{d_k}}\right)(P V) = P \, \text{Attention}(X)$$
  - Đây là tính chất **tương đương hoán vị (Permutation Equivariance)**. Khi lấy trung bình cộng các hàng qua tầng Mean Pooling ($\frac{1}{T}\sum_{t=1}^T z_t$), phép cộng có tính giao hoán nên hai câu ngược nghĩa như *"dog bites man"* và *"man bites dog"* sẽ sinh ra cùng một vector đại diện câu giống hệt nhau (bất biến hoán vị — **Permutation Invariance**).
- **Mức độ tự tin (Confidence):** 100% (Bảo đảm bằng đại số tuyến tính).

---

### Prediction 4: Sự đa dạng của các Attention Heads
- **Câu hỏi:** Hai attention heads có nhất thiết học cùng một attention pattern không?
- **Dự đoán (Prediction):** **KHÔNG.** Hai attention heads không nhất thiết học cùng một attention pattern; mỗi head thường tập trung vào các kiểu tương tác từ vựng khác nhau, dù đôi khi một số heads có thể chồng lấp một phần.
- **Lý do (Reason):**  
  1. Mỗi head $i \in \{1, \dots, h\}$ sở hữu bộ ma trận trọng số chiếu độc lập $(W_i^Q, W_i^K, W_i^V)$ được khởi tạo ngẫu nhiên khác nhau và chiếu đầu vào xuống các không gian con $\mathbb{R}^{d_k}$ khác nhau.
  2. Trong quá trình tối ưu hóa bằngݶ gradient descent, các heads có xu hướng phân tách vai trò để nắm bắt nhiều khía cạnh của câu (ví dụ: một head chú ý mạnh vào tính từ cảm xúc, một head khác chú ý vào trợ động từ phủ định hoặc cấu trúc lân cận).
- **Mức độ tự tin (Confidence):** 95% (Dựa trên cơ chế chiếu đa không gian con của Multi-Head Attention).

---

### Prediction 5: Tăng số lượng Attention Heads và Accuracy
- **Câu hỏi:** Tăng số attention heads có chắc chắn làm accuracy tăng không?
- **Dự đoán (Prediction):** **KHÔNG CHẮC CHẮN.** Tăng số lượng attention heads không đảm bảo accuracy sẽ tăng đơn điệu; khi số heads quá lớn, accuracy có thể bão hòa hoặc giảm nhẹ.
- **Lý do (Reason):**  
  - Khi giữ cố định chiều biểu diễn của mô hình $d_{\text{model}} = 128$, việc tăng số lượng heads $n_{\text{heads}} \in \{1, 2, 4, 8\}$ sẽ làm giảm số chiều của mỗi head theo tỉ lệ nghịch:
    $$d_k = \frac{d_{\text{model}}}{n_{\text{heads}}} \in \{128, 64, 32, 16\}$$
  - Tổng số lượng tham số của khối Multi-Head Attention hoàn toàn không đổi ($4 d_{\text{model}}^2 + 4 d_{\text{model}}$). Nếu chia thành quá nhiều heads khiến $d_k$ quá nhỏ, mỗi không gian con có thể không đủ dung lượng chiều để biểu diễn chính xác tương quan ngữ nghĩa phức tạp.
- **Mức độ tự tin (Confidence):** 90% (Dựa trên sự đánh đổi giữa số lượng heads $n_{\text{heads}}$ và chiều không gian con $d_k$).

---

### Prediction 6: Tốc độ của Transformer so với RNN trên mọi Sequence Length
- **Câu hỏi:** Transformer có luôn nhanh hơn RNN trên mọi sequence length không?
- **Dự đoán (Prediction):** **KHÔNG.** Transformer nhanh hơn RNN trên các chuỗi ngắn và trung bình nhờ khả năng tính toán song song, nhưng khi sequence length $T$ tăng lên rất lớn, Transformer sẽ trở nên chậm hơn và tiêu tốn bộ nhớ gấp nhiều lần so với RNN.
- **Lý do (Reason):**  
  - Độ phức tạp tính toán mỗi tầng của RNN/LSTM tăng tuyến tính theo độ dài chuỗi: $\mathcal{O}(T \cdot d_{\text{model}}^2)$.
  - Độ phức tạp của Self-Attention trong Transformer tăng theo hàm bậc hai của độ dài chuỗi: $\mathcal{O}(T^2 \cdot d_{\text{model}})$.
  - Khi độ dài chuỗi $T$ tăng gấp đôi ($128 \to 256 \to 512 \to 1024$), kích thước ma trận Attention $T \times T$ tăng gấp 4 lần ở mỗi bước ($16.384 \to 65.536 \to 262.144 \to 1.048.576$ phần tử mỗi head), khiến chi phí bậc hai $\mathcal{O}(T^2)$ vượt qua chi phí tuần tự tuyến tính $\mathcal{O}(T)$ của RNN.
- **Mức độ tự tin (Confidence):** 95% (Độ phức tạp thuật toán $\mathcal{O}(T^2)$ so với $\mathcal{O}(T)$).

---

## Kiểm chứng dự đoán sau thực nghiệm (Mục 28 — Prediction Verification)

| Prediction | Dự đoán ban đầu | Kết quả thực nghiệm đo đạc | Đúng/Sai | Giải thích dựa trên bằng chứng thực nghiệm |
|:---:|:---|:---|:---:|:---|
| **1** | **Có** (Transformer song song hơn RNN) | Ở `max_length = 128` (4 epochs), Transformer hoàn tất huấn luyện trong **46.90s** (Accuracy `79.20%`), nhanh gần gấp **2 lần** so với LSTM (**91.77s**, Accuracy `68.80%`). | **Đúng** | Phép nhân ma trận song song trên toàn chuỗi $T=128$ giúp Transformer loại bỏ vòng lặp tuần tự 128 bước thời gian của cổng LSTM. |
| **2** | **Không** (Bỏ PE sẽ mất thứ tự token) | Khi không có PE, vector đầu ra của từ `"dog"` ở vị trí chủ ngữ (`S1[0]`) và vị trí tân ngữ (`S2[2]`) có khoảng cách L2 bằng **`0.0000`**. Khi có PE, khoảng cách L2 của từ `"dog"` giữa 2 vị trí là **`0.8263`**. | **Đúng** | Không có PE, điểm số Attention giữa mọi cặp từ chỉ phụ thuộc vào nội dung vector embedding mà hoàn toàn mù vị trí. |
| **3** | **Không** (Không phân biệt được 2 câu hoán vị sau pooling) | Với `S1 = "dog bites man"` và `S2 = "man bites dog"`:<br>- **Exp A (Không PE):** Khoảng cách L2 giữa hai vector câu sau Mean Pooling = **`0.00000000`**.<br>- **Exp B (Có PE):** Khoảng cách L2 giữa hai câu = **`0.662406`**. | **Đúng** | Chứng minh thực nghiệm tính bất biến hoán vị (Permutation Invariance) của Self-Attention khi thiếu PE và vai trò phá vỡ đối xứng của Sinusoidal PE. |
| **4** | **Không nhất thiết** (Các heads học pattern khác nhau) | Trong Experiment 3 (`n_heads = 4`):<br>- **Head 1** tập trung nhọn vào `'great'` (`Entropy = 1.33`, `Max = 0.999`).<br>- **Head 2** chú ý vào `'not' -> 'ending'` (`Entropy = 2.36`).<br>- **Head 3** phân tán trên `'movie' -> 'good'` (`Entropy = 2.39`).<br>- **Head 4** chú ý vào `'the' -> 'truly'` (`Entropy = 2.37`). | **Đúng** | Cosine similarity giữa Head 1 và Head 2 chỉ là `0.3087`, chứng tỏ các heads học các phân phối chú ý khác nhau (dù Head 2 và Head 3 có chồng lấp một phần `0.7854`). |
| **5** | **Không chắc chắn** (Tăng heads không luôn tăng accuracy) | Giữ cố định $d_{\text{model}} = 128$:<br>- `1 head` ($d_k=128$): Acc = `0.7900`<br>- `2 heads` ($d_k=64$): Acc = **`0.7980`**<br>- `4 heads` ($d_k=32$): Acc = `0.7920`<br>- `8 heads` ($d_k=16$): Acc = `0.7830` | **Đúng** | Accuracy đạt đỉnh ở `n_heads = 2` (`79.80%`) và giảm nhẹ xuống `78.30%` ở `n_heads = 8` khi chiều mỗi head bị thu hẹp còn $d_k = 16$. |
| **6** | **Không** (Không luôn nhanh hơn RNN ở mọi độ dài chuỗi) | - Khi `max_length` tăng $32 \to 64 \to 128 \to 256$, thời gian train của Transformer tăng từ `10.90s -> 21.51s -> 46.90s -> 166.22s`.<br>- Ở benchmark $T=1024$ (Batch=16), Forward pass của Transformer mất **`194.58 ms`**, chậm hơn **6 lần** so với LSTM (**`32.52 ms`**). | **Đúng** | Khi chuỗi dài, chi phí bậc hai $\mathcal{O}(T^2)$ của ma trận Attention ($1024^2 = 1.048.576$ phần tử/head) áp đảo chi phí tuyến tính $\mathcal{O}(T)$ của LSTM. |
