# LAB 06: PREDICTION TRƯỚC EXPERIMENT

---

### Prediction 1: Key có Dot-Product lớn hơn các Key khác
- **Câu hỏi:** Nếu một key có dot product lớn hơn các key khác, attention weight của nó sẽ như thế nào?
- **Dự đoán (Prediction):** Token/Key đó sẽ nhận được attention weight lớn nhất trong toàn bộ hàng (sequence).
- **Lý do (Reason):** Trọng số Attention được tính qua hàm Softmax:  
  $$\alpha_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$$  
  Vì hàm mũ $e^x$ là hàm số đồng biến nghiêm ngặt trên $\mathbb{R}$ ($x_1 > x_2 \iff e^{x_1} > e^{x_2}$), nên phần tử có điểm số $s_i$ cao nhất chắc chắn sẽ có tử số lớn nhất trong khi mẫu số chung là tổng các số hạng dương cố định.
- **Mức độ tự tin (Confidence):** 100% (Bảo đảm bằng công thức toán học).

---

### Prediction 2: Tất cả Scores bằng nhau
- **Câu hỏi:** Nếu tất cả scores giống nhau $s = [1, 1, 1, 1]$, attention weights sẽ như thế nào?
- **Dự đoán (Prediction):** Phân phối attention weights sẽ là phân phối đều (uniform distribution), mỗi token nhận trọng số bằng nhau: $\alpha_i = \frac{1}{4} = 0.25$.
- **Lý do (Reason):** Nếu $s_1 = s_2 = s_3 = s_4 = c$, thì $\exp(s_i) = e^c$.  
  Mẫu số là $\sum_{j=1}^4 e^c = 4e^c$.  
  Khi đó $\alpha_i = \frac{e^c}{4e^c} = \frac{1}{4} = 0.25$ với mọi $i$. Khi các điểm số không có sự phân hóa, mô hình không có căn cứ để ưu tiên token nào hơn token nào.
- **Mức độ tự tin (Confidence):** 100% (Bảo đảm bằng toán học).

---

### Prediction 3: Một Score lớn hơn rất nhiều so với các Score khác
- **Câu hỏi:** Nếu một score lớn hơn rất nhiều $s = [1, 1, 10]$, attention sẽ tập trung như thế nào?
- **Dự đoán (Prediction):** Phân phối attention sẽ tập trung cực độ (cực nhọn - sharp/peaked) vào vị trí có score bằng 10, giá trị weight tại đó tiến sát tới 1.0 (gần như phân phối Dirac/one-hot), các vị trí còn lại xấp xỉ 0.
- **Lý do (Reason):** Do tính chất tăng theo hàm mũ của $e^x$:  
  $e^1 \approx 2.718$, trong khi $e^{10} \approx 22026.46$.  
  Tổng mẫu số: $2 \times 2.718 + 22026.46 \approx 22031.90$.  
  Trọng số tại score 10: $\frac{22026.46}{22031.90} \approx 0.99975$ ($\approx 99.98\%$), trong khi hai phần tử đầu chỉ chiếm xấp xỉ $0.012\%$.
- **Mức độ tự tin (Confidence):** 100% (Đặc tính tăng trưởng hàm mũ).

---

### Prediction 4: Chi phí tính toán khi độ dài chuỗi tăng từ 10 lên 1000 tokens
- **Câu hỏi:** Nếu sequence tăng từ 10 tokens $\to$ 1000 tokens, chi phí tính attention sẽ thay đổi như thế nào?
- **Dự đoán (Prediction):** Kích thước ma trận Attention $T \times T$ và số phép tính nhân vô hướng giữa mọi cặp token sẽ tăng gấp $10.000$ lần (tăng theo bậc hai $O(T^2)$), chứ không tăng tuyến tính 100 lần.
- **Lý do (Reason):**  
  Với chuỗi $T$ tokens, phép nhân ma trận $Q K^T$ (với $Q, K \in \mathbb{R}^{T \times d_k}$) sinh ra ma trận Attention Matrix kích thước $T \times T$.  
  - Khi $T = 10$: Ma trận có $10^2 = 100$ phần tử.  
  - Khi $T = 1000$: Ma trận có $1000^2 = 1.000.000$ phần tử.  
  - Tỉ lệ tăng: $\frac{1.000.000}{100} = 10.000$ lần (tương ứng $(\frac{1000}{10})^2 = 100^2$).
- **Mức độ tự tin (Confidence):** 100% (Độ phức tạp thuật toán $O(T^2)$).

---

### Prediction 5: Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?
- **Câu hỏi:** Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?
- **Dự đoán (Prediction):** **KHÔNG.** Attention weights không đồng nhất với trực giác ngôn ngữ học (linguistic importance) hay suy luận logic (reasoning) của con người.
- **Lý do (Reason):**  
  1. Trọng số Attention được tối ưu hóa thuần túy qua mục tiêu cực tiểu hàm mất mát (loss function) trong không gian biểu diễn toán học, không được huấn luyện dựa trên nhãn chú ý từ ngữ của con người.
  2. Attention phân tán và chịu ảnh hưởng mạnh của tần suất từ, vị trí từ, cấu trúc câu, và các hư từ (punctuation, stopwords) có thể vô tình hấp thụ lượng attention lớn làm "bồn chứa attention" (attention sinks).
  3. Nhiều công trình nghiên cứu nổi tiếng trong cộng đồng NLP (ví dụ: Jain & Wallace 2019 *"Attention is not Explanation"*, Wiegreffe & Pinter 2019 *"Attention is not not Explanation"*) đã chứng minh mối quan hệ giữa attention weights và khả năng giải thích dự đoán là rất phức tạp và không thể suy diễn một chiều.
- **Mức độ tự tin (Confidence):** 95% (Dựa trên lý thuyết học máy và các bài báo khoa học chuẩn mực).
