# LAB 09: DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PREDICTION) & ĐỐI CHIẾU THỰC TẾ

---

## PHẦN I: GIẢ THUYẾT NGHIÊN CỨU TIÊN NGHIỆM (PRE-EXPERIMENT PREDICTIONS)

### Prediction 1: Pretrained Transformer vs Simple Feature Baseline
* **Câu hỏi nghiên cứu:**  
  Một mô hình Pretrained Transformer (BERT Token Classification) có vượt trội hơn mô hình Baseline phân loại truyền thống dựa trên đặc trưng từ vựng cục bộ (Lexical Features + Naive Bayes/Linear Model) trên tập dữ liệu CoNLL-2003 không? Khoảng cách chênh lệch F1-score và Precision dự kiến là bao nhiêu?
* **Dự đoán (Prediction):**  
  **CÓ, VƯỢT TRỘI RÕ RỆT.** Pretrained Transformer sẽ vượt mô hình Baseline đặc biệt ở **Strict Precision** và F1-score tổng thể. Mô hình Baseline do chỉ nhìn vào các cửa sổ từ cục bộ $\pm 1$ sẽ bị nhiễu và dự đoán nhầm rất nhiều từ thường viết hoa hoặc danh xưng thành thực thể (Precision thấp). Ngược lại, cơ chế Self-Attention của BERT nắm bắt toàn bộ cú pháp câu, giúp lọc bỏ các nhận định sai lầm, nâng cao độ chính xác nghiêm ngặt (Strict Precision).
* **Lý do khoa học (Scientific Rationale):**  
  1. *Hạn chế của mô hình Baseline:* Mô hình trích xuất đặc trưng truyền thống dựa vào các biểu diễn bề mặt cục bộ (cửa sổ trượt $\pm 1$ từ, tiền tố, hậu tố, viết hoa). Tuy nhiên, đặc trưng từ vựng rời rạc (Sparse Handcrafted Features) không thể nắm bắt được sự tương tác ngữ nghĩa dài giữa các thành phần cách xa nhau trong câu và hoàn toàn thất bại khi gặp từ vựng ngoài tập huấn luyện (Out-of-Vocabulary - OOV).
  2. *Sức mạnh của Pretrained Transformer:* BERT sử dụng cơ chế Self-Attention hai chiều (Bidirectional Attention), cho phép mỗi token tính toán trực tiếp sự phụ thuộc với tất cả các từ khác trong câu. Hơn nữa, quá trình tiền huấn luyện trên hàng tỷ từ giúp BERT tích lũy tri thức sâu sắc về cấu trúc ngữ pháp và vai trò cú pháp.
* **Mức độ tự tin (Confidence):** **95%** (Dựa trên bản chất vượt trội của cơ chế Self-Attention đối với bài toán phụ thuộc ngữ cảnh).

---

### Prediction 2: Độ dài Thực thể và Ranh giới (Single-token vs Multi-token Entities)
* **Câu hỏi nghiên cứu:**  
  Mô hình có đạt F1-score cao hơn rõ rệt trên các thực thể ngắn so với các thực thể dài (Multi-token entities: 3, 4+ tokens) không?
* **Dự đoán (Prediction):**  
  **CÓ, F1-SCORE SẼ SUY GIẢM RẤT MẠNH KHI ĐỘ DÀI THỰC THỂ ĐẠT TỪ 3 TOKENS TRỞ LÊN.** F1-score của thực thể 1-2 tokens sẽ cao nhất, sau đó sụt giảm nghiêm trọng ở các thực thể 3 tokens và thấp nhất ở các thực thể từ 4 tokens trở lên.
* **Lý do khoa học (Scientific Rationale):**  
  1. *Định nghĩa đánh giá nghiêm ngặt (Strict Entity-level Evaluation của Seqeval):* Một thực thể chỉ được coi là True Positive khi và chỉ khi **toàn bộ span** được dự đoán chính xác cả về ranh giới bắt đầu (Start Index), ranh giới kết thúc (End Index), lẫn loại thực thể (Entity Type). Bất kỳ một sai sót nào ở một token con bên trong đều biến toàn bộ thực thể đó thành một False Negative (bỏ sót) đồng thời tạo ra một False Positive (dự đoán sai).
  2. *Quy tắc nhân xác suất chuỗi:* Với một thực thể có độ dài $L$ tokens $E = (t_1, t_2, \dots, t_L)$, xác suất dự đoán chính xác toàn bộ thực thể xấp xỉ tích xác suất của từng token thành phần:
     $$P(\text{Entity correct}) = P(y_1 = \text{B-Type} \mid \mathbf{x}) \prod_{k=2}^L P(y_k = \text{I-Type} \mid \mathbf{x})$$
     Vì mỗi xác suất thành phần đều nhỏ hơn 1 ($P_k < 1$), khi chiều dài $L \ge 3$, xác suất xảy ra lỗi ranh giới (Boundary Drift, gán nhầm token giữa chừng hoặc ngắt sớm) tăng lên theo cấp số nhân.
* **Mức độ tự tin (Confidence):** **90%** (Dựa trên lý thuyết xác suất chuỗi và tiêu chuẩn chấm điểm khắt khe của Seqeval).

---

### Prediction 3: Tần suất Thực thể trong Tập Huấn luyện & Tỷ lệ Lỗi (Rare Entities & False Negatives)
* **Câu hỏi nghiên cứu:**  
  Mối tương quan giữa tần suất xuất hiện của thực thể trong tập Train và tỷ lệ lỗi dự đoán (đặc biệt là lỗi bỏ sót False Negatives) trên tập Test như thế nào?
* **Dự đoán (Prediction):**  
  **TỒN TẠI MỐI TƯƠNG QUAN NGHỊCH ĐẢO MẠNH MẼ.** Các thực thể xuất hiện hiếm gặp (tần suất $\le 2$ lần trong tập Train) hoặc tên riêng nước ngoài hoàn toàn mới lạ đối với tập Test sẽ có tỷ lệ False Negative cao hơn ít nhất gấp 2 đến 3 lần so với các thực thể phổ biến (xuất hiện $> 10$ lần trong tập Train).
* **Lý do khoa học (Scientific Rationale):**  
  1. Mặc dù Pretrained Transformer có vốn từ vựng con (WordPiece) phong phú và khả năng tổng quát hóa ngữ cảnh tốt, các thực thể hiếm thường bị phân tách thành chuỗi nhiều subword ít phổ biến.
  2. Khi một tên riêng hiếm gặp xuất hiện trong cấu trúc câu trung tính (ngữ cảnh không cung cấp manh mối phân loại quá rõ ràng), mô hình có xu hướng thiên kiến quy nạp (Inductive Bias) về nhãn đa số `O` (nhãn ngoài chiếm hơn $80\%$ tổng số tokens trong tập dữ liệu CoNLL-2003). Do đó, các thực thể hiếm sẽ dễ bị mô hình "lờ đi" và gán nhãn `O` (False Negative).
* **Mức độ tự tin (Confidence):** **85%** (Dựa trên hiện tượng mất cân bằng nhãn trầm trọng trong bài toán Token Classification).

---

## PHẦN II: BẢNG ĐỐI CHIẾU KIỂM CHỨNG SAU THỰC NGHIỆM (POST-EXPERIMENT VERIFICATION)

| Dự đoán | Giả thuyết Tiên nghiệm | Bằng chứng Thực nghiệm Đo đạc Thực tế | Kết luận | Phân tích Nguyên nhân Kỹ thuật |
|:---:|:---|:---|:---:|:---|
| **Prediction 1**<br>*(Transformer vs Baseline)* | Transformer vượt trội Baseline rõ rệt, đặc biệt ở Strict Precision và F1 tổng thể. | - **Baseline:** F1 = `0.6477`, Prec = `0.5742`, Rec = `0.7427`<br>- **BERT-Tiny:** **F1 = `0.6927`**, **Prec = `0.6730`**, Rec = `0.7136`<br>- Strict Precision tăng vọt: **+9.88%** (`0.6730` vs `0.5742`)<br>- Tên người (`PER`) F1: **`0.7974`** vs `0.6693` (**+12.81%**) | **ĐÚNG** | Baseline gặp hiện tượng False Positives cực lớn do thiếu hiểu biết ngữ cảnh dài (Precision chỉ đạt $57.42\%$). BERT với cơ chế Bidirectional Attention phân tích chính xác vai trò cú pháp của từ, giúp loại bỏ các dự đoán nhầm và tăng vọt độ chuẩn xác ($67.30\%$). |
| **Prediction 2**<br>*(Độ dài thực thể & Ranh giới)* | F1-score sụt giảm mạnh khi độ dài thực thể tăng (đặc biệt từ 3 tokens trở lên). | - **Length 1 (3,573 mẫu):** F1 = **`0.6820`** (P: 0.6168, R: 0.7627)<br>- **Length 2 (1,775 mẫu):** F1 = **`0.7630`** (P: 0.8557, R: 0.6885)<br>- **Length 3 (236 mẫu):** F1 = **`0.4314`** (P: 0.6364, R: 0.3263)<br>- **Length 4+ (62 mẫu):** F1 = **`0.1220`** (P: 0.2500, R: 0.0806) | **ĐÚNG** | F1 giữ mức cao ở độ dài 1-2 tokens (tên người, địa danh ngắn), nhưng **sụp đổ từ `0.7630` xuống `0.4314` (giảm 43.5%) ở độ dài 3** và **rơi xuống `0.1220` ở độ dài 4+**. Lỗi ranh giới phân mảnh (Boundary Drift) khiến toàn bộ span dài bị `seqeval` đánh trượt. |
| **Prediction 3**<br>*(Thực thể hiếm & False Negatives)* | Thực thể hiếm gặp có tỷ lệ False Negative cao gấp 2-3 lần thực thể phổ biến. | - **Thực thể phổ biến** (như Japan, London, England): Tỷ lệ nhận diện đúng đạt $> 85\%$.<br>- **Thực thể hiếm / Tên riêng lạ** (như Bitar, Cuttitta, Coste, Bordon): Bị mô hình bỏ sót hoàn toàn và gán nhãn `O` (False Negatives chiếm $> 35\%$). | **ĐÚNG** | Tỷ lệ nhãn `O` chiếm ưu thế áp đảo trong tập dữ liệu (~83%) tạo ra thiên kiến an toàn; khi gặp từ hiếm bị chia nhỏ subwords trong cấu trúc trung tính, mô hình ưu tiên dự đoán `O` thay vì mạo hiểm gán nhãn thực thể. |
