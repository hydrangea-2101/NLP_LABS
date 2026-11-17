# LAB 09: BÁO CÁO PHẢN TƯ / ĐÚC KẾT HỌC TẬP (REFLECTION)

---

### 1. Sự thống nhất kiến trúc của Transformer: Từ Sequence Classification đến Token Classification

Một trong những phát hiện sâu sắc và ấn tượng nhất trong chuỗi bài thực hành NLP là **sự thống nhất tuyệt đối về mặt kiến trúc** của mô hình Transformer tiền huấn luyện (Pretrained Transformer):
* Ở **LAB 08 (Sequence Classification — Phân loại cảm xúc IMDb)**: Bài toán yêu cầu ánh xạ toàn bộ văn bản $X = (x_1, \dots, x_T)$ sang một nhãn vô hướng duy nhất $y \in \{0, 1\}$. Chúng ta chỉ cần trích xuất vector biểu diễn trạng thái ẩn tại token đặc biệt đầu tiên $\mathbf{h}_{\text{[CLS]}} \in \mathbb{R}^d$ và đưa qua một tầng Linear Head:
  $$P(y \mid X) = \text{Softmax}(\mathbf{h}_{\text{[CLS]}} W_{\text{seq}}^T + b_{\text{seq}})$$
* Sang **LAB 09 (Token Classification — Nhận diện thực thể CoNLL-2003)**: Bài toán chuyển dịch thành việc gán nhãn cho từng phần tử $X \to Y = (y_1, \dots, y_T)$ với $y_t \in \{1, \dots, C\}$. Toàn bộ phần thân của mạng tiền huấn luyện (Pretrained Encoder Backbone gồm các khối Multi-Head Self-Attention, Feed-Forward, LayerNorm) **được giữ nguyên vẹn 100%**. Điểm thay đổi duy nhất là thay thế Classifier Head: thay vì nén về một token, ta áp dụng ma trận chiếu $W_{\text{token}} \in \mathbb{R}^{C \times d}$ lên **toàn bộ ma trận trạng thái ẩn** $H \in \mathbb{R}^{T \times d}$:
  $$Z = H W_{\text{token}}^T + b_{\text{token}} \in \mathbb{R}^{T \times C}$$

Khả năng này minh chứng rằng cơ chế Self-Attention hai chiều của BERT không chỉ học được biểu diễn tổng thể của văn bản mà còn duy trì một trường biểu diễn cục bộ cực kỳ giàu thông tin ngữ cảnh cho từng vị trí token.

---

### 2. Đánh đổi giữa Token-Level Classification và Span-Based / Seq2Seq Extraction

Trong bài toán trích xuất thực thể và thông tin (Information Extraction), phương pháp gán nhãn token (Token Classification qua BIO scheme) sở hữu những ưu điểm và hạn chế rõ rệt khi đặt cạnh các hướng tiếp cận khác:
1. **Token Classification (Sequence Labeling):**
   * *Ưu điểm:* Tốc độ suy luận cực nhanh ($\mathcal{O}(T)$ sau encoder), tính toán song song hoàn toàn, dễ huấn luyện bằng Cross-Entropy độc lập trên từng vị trí.
   * *Nhược điểm:* Giả định độc lập có điều kiện giữa các nhãn (Conditional Independence) khiến mô hình có nguy cơ sinh ra chuỗi nhãn bất hợp lệ (ví dụ: `O` đi liền với `I-PER` mà không có `B-PER` mở đầu, hoặc `B-LOC` nhảy sang `I-ORG`).
2. **Span-based Approaches (Phân loại đoạn Span):**
   * Mô hình liệt kê các cặp chỉ số $(i, j)$ khả dĩ và phân loại trực tiếp toàn bộ span $[i, j]$. Cách tiếp cận này giải quyết triệt để vấn đề thực thể lồng nhau (Nested Entities), nhưng chi phí tính toán tăng lên bậc hai $\mathcal{O}(T^2)$.
3. **Generative / Seq2Seq Extraction (T5, GPT):**
   * Mô hình sinh trực tiếp chuỗi JSON hoặc văn bản có cấu trúc. Rất linh hoạt và khái quát hóa mạnh, nhưng tốc độ sinh tự hồi quy (Autoregressive) chậm hơn gấp hàng chục lần so với Token Classification và tiềm ẩn nguy cơ ảo giác (Hallucination).

---

### 3. Tác động của Subword Tokenization và Chiến lược Canh chỉnh Nhãn (Strategy A)

Việc sử dụng thuật toán tách từ con WordPiece đặt ra một thách thức cố hữu: **sự không tương thích ranh giới (Boundary Mismatch)** giữa nhãn thực thể cấp từ (Word-level) và chuỗi tokens cấp từ con (Subword-level).
* **Strategy A (First-subword alignment với `ignore_index = -100`):**
  Bằng cách chỉ gán nhãn cho subword đầu tiên và gán `-100` cho tất cả các mảnh subword tiếp theo (`##...`) cũng như các token đặc biệt (`[CLS]`, `[SEP]`, `[PAD]`), chúng ta đảm bảo hàm mất mát Cross-Entropy không bị thiên lệch bởi những từ dài bị bẻ thành nhiều mảnh. Đạo hàm riêng tại các vị trí `-100` triệt tiêu về 0, giúp gradient tập trung chính xác vào việc điều chỉnh vector khởi đầu mang tính định hình của từ.
* Thực nghiệm trong notebook `tokenizer_alignment.ipynb` đã chứng minh tỷ lệ giãn nở subword $R = \frac{\#\text{subwords}}{\#\text{words}}$ đạt trung bình $\approx 1.15$. Nếu không có cơ chế masking `-100` chuẩn xác, việc sao chép nhãn bừa bãi cho mọi mảnh hậu tố (Strategy B) sẽ gây nhiễu tín hiệu học và làm sai lệch chỉ số đánh giá `seqeval`.

---

### 4. Mối liên hệ giữa Pretrained Representations và Nhiệm vụ Cấp Token Tinh vi

Quá trình tiền huấn luyện Masked Language Modeling (MLM) của BERT buộc mô hình phải phục hồi một từ bị che dựa trên toàn bộ ngữ cảnh hai chiều xung quanh. Nhiệm vụ này tình cờ chuẩn bị một không gian đặc trưng lý tưởng cho Token Classification:
* Mỗi trạng thái ẩn $h_t$ không chỉ mã hóa ngữ nghĩa nội tại của từ $t$, mà còn mã hóa vai trò ngữ pháp, cấu trúc danh ngữ, và quan hệ lệ thuộc với các động từ/giới từ lân cận.
* Khi fine-tuning trên CoNLL-2003, mô hình không cần phải học lại xem "founded", "in", "president" đóng vai trò gì. Thay vào đó, nó tận dụng trực tiếp các mẫu ngữ cảnh đã học để nhận diện chính xác các thực thể trừu tượng phức tạp (`ORG`, `MISC`), tạo ra bước nhảy vọt hiệu năng hơn $20\%$ F1 so với mô hình Baseline dựa trên đặc trưng từ vựng cục bộ.
