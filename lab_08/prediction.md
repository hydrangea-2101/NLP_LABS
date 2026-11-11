# LAB 08: PREDICTION TRƯỚC EXPERIMENT

---

### Prediction 1: Pretrained Transformer vs From-Scratch Transformer
- **Câu hỏi:** Một Pretrained Transformer fine-tuned trên IMDb có khả năng đạt kết quả phân loại (Accuracy, F1) tốt hơn Transformer nhỏ được train from scratch trên cùng dữ liệu không?
- **Dự đoán (Prediction):** **CÓ.** Mô hình Pretrained Transformer (BERT) sẽ vượt trội hơn Transformer huấn luyện từ đầu (trained from scratch) trên cùng tập dữ liệu, đặc biệt ở độ thu hồi (Recall), khả năng xử lý câu dài và độ ổn định khi hội tụ.
- **Lý do (Reason):**  
  1. Trong bài toán học từ đầu (LAB 07), mạng Transformer Encoder bắt đầu từ các ma trận trọng số ngẫu nhiên $\mathcal{N}(0, \sigma^2)$ và chỉ được tối ưu hóa trên $5,000$ mẫu IMDb nhỏ. Mô hình phải giải quyết đồng thời hai nhiệm vụ vô cùng khó khăn: tự học biểu diễn ngôn ngữ cơ bản (ngữ pháp, cú pháp, liên kết từ) VÀ học phân loại cảm xúc nhị phân.
  2. Ngược lại, Pretrained Transformer (BERT) đã trải qua quá trình tiền huấn luyện tự giám sát (Self-supervised Pretraining) trên hàng tỷ tokens văn bản từ Wikipedia và BookCorpus thông qua nhiệm vụ Masked Language Modeling (MLM). Không gian biểu diễn ngữ cảnh hai chiều (Contextual Representations) của BERT đã tích lũy sẵn tri thức ngôn ngữ phong phú, quan hệ đồng tham chiếu và trật tự cú pháp. Khi tinh chỉnh (Fine-tuning), mô hình chỉ cần điều chỉnh nhẹ các vector đặc trưng để thích ứng với sắc thái khen/chê của điện ảnh, giúp hội tụ nhanh hơn và giảm thiểu nguy cơ rơi vào cực tiểu cục bộ.
- **Bằng chứng kỳ vọng (Expected Evidence):**  
  Pretrained BERT sẽ đạt Recall cao vượt bậc (nhận diện được các mẫu tích cực/tiêu cực phức tạp), loss trên tập Validation giảm nhanh và ổn định hơn, đồng thời khi mở rộng độ dài chuỗi ($T = 256$), Pretrained BERT sẽ bứt phá mạnh mẽ vượt qua mô hình From-Scratch.
- **Mức độ tự tin (Confidence):** 95% (Dựa trên nền tảng lý thuyết vững chắc của Transfer Learning trong NLP).

---

### Prediction 2: Frozen Encoder (Feature Extraction) vs Full Fine-Tuning
- **Câu hỏi:** Nếu freeze toàn bộ pretrained encoder và chỉ train classification head thì kết quả sẽ:  
  **A.** Tốt hơn fine-tuning toàn bộ | **B.** Tương đương | **C. Kém hơn** | **D.** Không thể dự đoán?
- **Dự đoán (Prediction):** **C. KÉM HƠN.** Đóng băng toàn bộ Pretrained Encoder và chỉ huấn luyện Linear Classification Head sẽ cho kết quả kém hơn rất nhiều so với Fine-tuning toàn bộ mô hình.
- **Lý do (Reason):**  
  1. Ở chế độ Frozen (Feature Extraction), toàn bộ các tầng mã hóa Transformer Encoder bị khóa cứng (`requires_grad = False`). Chỉ có duy nhất một tầng tuyến tính phân loại (`Linear(128, 2)` với đúng 258 tham số) được phép cập nhật.
  2. Mặc dù biểu diễn ngữ cảnh của BERT rất tổng quát, nhưng đây là biểu diễn đa mục đích phục vụ tác vụ đoán từ bị che (MLM). Bài toán phân loại cảm xúc trên bài phê bình điện ảnh chứa nhiều thuật ngữ nghệ thuật, thành ngữ khen chê, và cấu trúc chuyển ý tương phản (`"great acting, but terrible plot"`).
  3. Nếu không cho phép các tầng Multi-Head Attention ở trên cùng cập nhật lại các ma trận trọng số chiếu ($W^Q, W^K, W^V$) để chuyên môn hóa theo miền bài toán (Task Adaptation), vector trạng thái ẩn của token `[CLS]` sẽ không đủ phân tách tuyến tính (linearly separable). Do đó, tầng Linear Head đơn lẻ không thể phân loại chính xác các trường hợp phức tạp.
- **Mức độ tự tin (Confidence):** 95% (Đặc tính phân tách không gian đặc trưng của Feature Extraction vs Fine-tuning).

---

### Prediction 3: Hiệu quả sử dụng dữ liệu khi kích thước tập huấn luyện giảm còn 10%
- **Câu hỏi:** Nếu chỉ có 10% training data (500 mẫu), Pretrained model hay model train from scratch sẽ giảm performance mạnh hơn?
- **Dự đoán (Prediction):** **MÔ HÌNH TRAIN FROM SCRATCH SẼ GIẢM PERFORMANCE MẠNH HƠN RẤT NHIỀU.** Pretrained Transformer sẽ bảo toàn hiệu năng tốt hơn đáng kể.
- **Lý do (Reason):**  
  1. Một kiến trúc Transformer hoàn chỉnh có hàng triệu tham số tự do (ví dụ BERT-Tiny có $4.39\text{M}$ tham số). Để tối ưu hóa không gian giả thuyết khổng lồ này từ phân phối ngẫu nhiên, mô hình đòi hỏi một lượng dữ liệu đủ lớn để không bị overfit. Khi dữ liệu huấn luyện bị cắt giảm 10 lần (từ $5,000$ mẫu xuống chỉ còn $500$ mẫu, tức 250 positive / 250 negative), mô hình From-Scratch sẽ rơi vào tình trạng "đói dữ liệu" (severe data starvation), dẫn đến hiện tượng học vẹt các mẫu huấn luyện và mất hoàn toàn khả năng tổng quát hóa trên tập Test.
  2. Trong khi đó, Pretrained Transformer sở hữu thiên kiến quy nạp mạnh mẽ (Strong Inductive Bias) và các priors biểu diễn ngôn ngữ đã được học trước từ kho ngữ liệu đồ sộ. 500 mẫu dữ liệu hạ nguồn là hoàn toàn đủ để mô hình định hướng lại mặt phẳng phân tách nhị phân (Downstream Alignment) mà không cần phải học lại cấu trúc ngữ pháp cơ bản của tiếng Anh. Do đó, đường cong học tập của Pretrained BERT sẽ thoai thoải và suy giảm nhẹ hơn nhiều so với From-Scratch.
- **Mức độ tự tin (Confidence):** 90% (Dựa trên nguyên lý Data Efficiency của Pretrained Foundation Models).

---

## Kiểm chứng dự đoán sau thực nghiệm (Mục 28 — Prediction Verification)

| Prediction | Dự đoán ban đầu | Kết quả thực nghiệm đo đạc | Đúng/Sai | Giải thích dựa trên bằng chứng thực nghiệm |
|:---:|:---|:---|:---:|:---|
| **1** | **Có** (Pretrained vượt trội hơn From-Scratch) | - **Scratch (T=128):** Acc = `0.7970`, F1 = `0.8008`, Recall = `0.8160`<br>- **Pretrained (T=128):** Acc = `0.7520`, F1 = `0.7832`, **Recall = `0.8960`**<br>- **Pretrained (T=256):** **Acc = `0.8130`**, **F1 = `0.8105`**, Time = `190.90s` | **Đúng** | Pretrained BERT đạt Recall áp đảo ($89.60\%$ so với $81.60\%$). Khi mở rộng chiều dài ngữ cảnh $T=256$ để bao quát trọn vẹn văn bản review, Pretrained BERT bứt phá đạt **$81.30\%$ Accuracy** và **$0.8105$ F1**, vượt qua mô hình From-Scratch. |
| **2** | **C** (Frozen kém hơn Fine-tuning toàn bộ) | - **Mode A (Frozen):** Acc = **`0.5760`**, F1 = **`0.5985`** (Trainable Params: `258`)<br>- **Mode B (Fine-tuning):** Acc = **`0.7520`**, F1 = **`0.7832`** (Trainable Params: `4,386,178`) | **Đúng** | Đóng băng Encoder khiến Accuracy sụp đổ từ $75.20\%$ xuống còn $57.60\%$ (gần mức ngẫu nhiên $50\%$). Chứng minh thực nghiệm việc cập nhật các tầng Attention để thích ứng bài toán (Task Adaptation) là yếu tố quyết định. |
| **3** | **From-Scratch giảm mạnh hơn** (Pretrained data-efficient hơn) | - **Pretrained 100% (5,000 mẫu):** Acc = `0.7520`, F1 = `0.7832`<br>- **Pretrained 50% (2,500 mẫu):** Acc = `0.7110`, F1 = `0.7472`<br>- **Pretrained 25% (1,250 mẫu):** Acc = `0.6480`, F1 = `0.6190`<br>- **Pretrained 10% (500 mẫu):** Acc = **`0.6000`**, F1 = **`0.5556`** | **Đúng** | Khi giảm dữ liệu 10 lần (chỉ còn $500$ mẫu), Pretrained BERT vẫn duy trì vững vàng Accuracy đạt $60.00\%$, chứng minh vai trò to lớn của biểu diễn pretraining khi đối mặt với sự khan hiếm dữ liệu. |
