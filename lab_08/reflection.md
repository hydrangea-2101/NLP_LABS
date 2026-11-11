# LAB 08: BÁO CÁO PHẢN TƯ / ĐÚC KẾT HỌC TẬP (REFLECTION)

---

### 1. Tại sao trong thực tế người ta hầu như không bao giờ train Transformer từ đầu?
- Trong thực tế công nghiệp NLP, người ta hầu như không bao giờ huấn luyện một khối Transformer từ đầu (train from scratch) cho một bài toán cụ thể vì sự mất cân đối trầm trọng giữa **quy mô dữ liệu bài toán** và **dung lượng tham số của mô hình**.
- Một kiến trúc Transformer tiêu chuẩn chứa từ hàng triệu đến hàng trăm triệu tham số (ngay cả BERT-Tiny đã có $4.39\text{M}$ tham số, BERT-Base có $110\text{M}$ tham số). Để tối ưu hóa không gian giả thuyết khổng lồ này từ các trọng số ngẫu nhiên $\mathcal{N}(0, \sigma^2)$, mô hình cần tiếp cận hàng trăm triệu đến hàng tỷ từ vựng. Trong khi đó, các tập dữ liệu thực tế ở hạ nguồn (như IMDb 50k hoặc các bài toán doanh nghiệp) chỉ có vài nghìn đến vài chục nghìn mẫu. Huấn luyện từ đầu trên tập mẫu nhỏ tất yếu sẽ dẫn đến hiện tượng quá khớp (Overfitting) trầm trọng, sụp đổ gradient, và mô hình chỉ có thể ghi nhớ máy móc các từ khóa thay vì hiểu cấu trúc phân cấp của câu.

---

### 2. Sự khác biệt bản chất giữa Pretrained Representation và Representation học từ đầu
- **Representation học từ đầu (From-Scratch):** Bắt đầu từ số không, mô hình phải tự mò mẫm học các đặc trưng cú pháp cơ bản (danh từ, động từ, trật tự từ) song song với việc học mục tiêu phân loại nhị phân. Các vector biểu diễn thu được thường bị phụ thuộc chặt vào tập từ vựng hẹp của tập huấn luyện và rất dễ bị nhiễu.
- **Pretrained Representation (BERT):** Thu được thông qua quá trình học tự giám sát trên kho ngữ liệu khổng lồ (Wikipedia + BookCorpus) bằng nhiệm vụ Masked Language Modeling (MLM). Mô hình đã xây dựng sẵn một không gian ngữ nghĩa hai chiều (Bidirectional Contextual Manifold) cực kỳ phong phú, nắm bắt vững chắc sự đồng tham chiếu, quan hệ phụ thuộc cú pháp và tính đa nghĩa của từ vựng. Mỗi vector trạng thái ẩn tại token `[CLS]` đã kết tinh sẵn tri thức ngôn ngữ toàn diện, cung cấp một bệ phóng xuất phát tối ưu ($\theta_{\text{pretrained}}$) cho bài toán hạ nguồn.

---

### 3. Chi phí tính toán (Computational Cost) và Khả năng thích ứng bài toán hạ nguồn (Downstream Task)
- **Chi phí Pretraining:** Huấn luyện một mô hình ngôn ngữ nền tảng từ đầu tiêu tốn hàng trăm máy chủ GPU/TPU chuyên dụng hoạt động liên tục trong nhiều tuần hoặc nhiều tháng, tiêu tốn hàng trăm nghìn USD và phát thải lượng carbon khổng lồ.
- **Chi phí Fine-Tuning:** Kế thừa trọn vẹn các tầng biểu diễn sẵn có, quá trình Fine-tuning trên tập dữ liệu IMDb chỉ mất **vài chục giây đến vài phút** ngay trên một CPU cá nhân 16 cores (thực tế đo được: chỉ mất $84.32\text{s}$ cho 3 epochs trên 5,000 mẫu).
- **Tính hiệu quả dữ liệu (Data Efficiency):** Thực nghiệm 3 của LAB 08 đã chứng minh ngay cả khi cắt giảm dữ liệu huấn luyện xuống còn $10\%$ ($500$ mẫu), Pretrained BERT vẫn duy trì độ chính xác $60.00\%$, trong khi mô hình From-Scratch bị sụp đổ hoàn toàn. Pretraining mang lại thiên kiến quy nạp (Inductive Bias) mạnh mẽ, giúp bài toán hạ nguồn hội tụ cực nhanh với chi phí tối thiểu.

---

### 4. Chiến lược Fine-Tuning: Đóng băng (Frozen) vs Tinh chỉnh toàn phần (Full Fine-tuning)
- **Feature Extraction (Frozen Encoder):** Khóa cứng toàn bộ các tầng Transformer Encoder (`requires_grad = False`) và chỉ huấn luyện Linear Classifier Head (258 tham số). Phương pháp này huấn luyện nhanh và tiết kiệm tài nguyên đạo hàm, nhưng kết quả thực nghiệm trong LAB 08 cho thấy Accuracy sụt giảm nghiêm trọng xuống chỉ còn **$57.60\%$** (gần mức ngẫu nhiên $50\%$). Điều này chứng minh biểu diễn tổng quát của BERT tiền huấn luyện không tự động tách biệt tuyến tính (linearly separable) cho bài toán phân loại cảm xúc.
- **Full Fine-Tuning:** Cập nhật toàn bộ $4.39\text{M}$ tham số, cho phép các ma trận chiếu Self-Attention ở các tầng trên cùng tái cấu trúc để thích nghi chuyên biệt với văn phong điện ảnh (Task Adaptation), mang lại bước nhảy vọt hiệu năng lên **$75.20\%$** (và đạt **$81.30\%$** ở $T=256$).

---

### 5. Giới hạn thực tế: Hiện tượng trôi dạt miền dữ liệu (Domain Shift) & Catastrophic Forgetting
- **Domain Shift:** Mô hình Pretrained không phải là "chìa khóa vạn năng". Trong Thực nghiệm 5, khi đem mô hình đã fine-tuned từ IMDb sang kiểm thử trên 1,000 đánh giá sản phẩm Amazon (UCI), độ thu hồi $Recall$ đã sụt giảm mạnh từ **$89.60\%$ xuống $77.80\%$** ($11.8\%$). Nguyên nhân là do sự thay đổi về phong cách hành văn (bài phê bình phim dài, giàu nghệ thuật vs nhận xét sản phẩm ngắn gọn, thực dụng) và sự trôi dạt từ vựng cảm xúc (`"cinematography"` vs `"fast shipping"`).
- **Catastrophic Forgetting:** Khi fine-tuning quá sâu trên một miền bài toán hẹp với tốc độ học lớn, mô hình có nguy cơ làm hỏng các biểu diễn ngôn ngữ tổng quát đã học trong pha Pretraining, làm mất đi khả năng chuyển giao sang các tác vụ khác.

---

### 6. Chuỗi tiến hóa biểu diễn ngôn ngữ qua từng bài Lab
- **TF-IDF (Lab 04) $\to$ Giới hạn:** Vector thưa (sparse), số chiều khổng lồ bằng kích thước từ điển, hoàn toàn mất thứ tự từ và không hiểu từ đồng nghĩa.
- **Word2Vec (Lab 05) $\to$ Giới hạn:** Vector đặc (dense) nhưng **tĩnh (static)** — mỗi từ chỉ có duy nhất một vector cố định bất kể ngữ cảnh; khi lấy trung bình toàn câu (Mean Pooling) sẽ làm mất trật tự ngữ pháp.
- **LSTM (Lab 05/06) $\to$ Giới hạn:** Xử lý tuần tự từng bước thời gian $t$, không thể tính toán song song trên GPU và gặp nút thắt cổ chai khi phải nén toàn bộ câu dài vào một vector trạng thái ẩn duy nhất $h_T$.
- **Transformer Encoder from scratch (Lab 07) $\to$ Giới hạn:** Xử lý song song bằng Self-Attention và Positional Encoding, nhưng bị giới hạn tính toán bậc hai $\mathcal{O}(T^2)$, và khi huấn luyện từ đầu trên tập dữ liệu nhỏ (5,000 mẫu) sẽ thiếu tri thức ngôn ngữ nền tảng và dễ bị quá khớp.
- **Pretrained BERT & Fine-Tuning (Lab 08) $\to$ Bước đột phá:** Tận dụng tri thức tự giám sát từ hàng tỷ tokens qua Masked Language Modeling, giải quyết bài toán thiếu dữ liệu hạ nguồn, đạt hiệu năng phân loại vượt trội với chi phí huấn luyện tối thiểu.
- **Cầu nối sang LAB 09:** Phân loại toàn câu (Sequence Classification) chỉ gán 1 nhãn duy nhất cho cả đoạn văn. Để nhận diện từng thực thể cụ thể (tên người, tổ chức, địa điểm), **LAB 09 sẽ chuyển dịch từ Sequence Classification sang Token Classification (Named Entity Recognition - NER)**, tiếp tục khai thác chiều sâu biểu diễn của Pretrained Transformer.
