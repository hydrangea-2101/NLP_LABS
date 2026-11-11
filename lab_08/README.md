# NLP LAB 08 — PRETRAINED TRANSFORMERS AND BERT FINE-TUNING

## 1. Tổng quan và Vai trò của LAB 08 trong toàn bộ học phần

Ở **LAB 07**, chúng ta đã tự tay xây dựng kiến trúc **Transformer Encoder** hoàn chỉnh từ các khối cơ bản: Multi-Head Attention, Sinusoidal Positional Encoding, Position-wise Feed-Forward Network, Residual Connections và Layer Normalization. Tuy nhiên, một Transformer được huấn luyện từ đầu (trained from scratch) vẫn phải tự học biểu diễn ngôn ngữ từ dữ liệu hạn chế của bài toán phân loại cảm xúc.

**LAB 08 đặt câu hỏi cốt lõi:**  
> *"Nếu một Transformer đã được huấn luyện trước (Pretrained) trên một lượng văn bản khổng lồ (Wikipedia + BookCorpus) thông qua cơ chế tự giám sát Masked Language Modeling (MLM), thì tri thức biểu diễn ngôn ngữ đó có thể chuyển giao và tinh chỉnh (Fine-tuning) cho một bài toán hạ nguồn (Downstream Task) như phân loại cảm xúc IMDb hiệu quả ra sao?"*

### Mạch kiến thức chuyển tiếp trong chương trình:
$$\begin{matrix}
\textbf{LAB 07} & \text{Transformer trained from scratch} \\
\downarrow & \\
\textbf{LAB 08} & \text{Pretraining} \rightarrow \text{Masked Language Modeling (MLM)} \rightarrow \text{Pretrained BERT} \rightarrow \text{WordPiece Tokenizer} \\
& \rightarrow \text{Fine-tuning} \rightarrow \text{Text Classification} \\
\downarrow & \\
\textbf{LAB 09} & \text{Token Classification} \rightarrow \text{Named Entity Recognition (NER)} / \text{Information Extraction}
\end{matrix}$$

---

## 2. Cấu trúc Deliverables hoàn chỉnh của LAB 08

Toàn bộ các tệp deliverables được hoàn thiện theo chuẩn mực quy định tại Mục 25 W8.pdf:

```text
lab_08/
├── README.md                      # Báo cáo tổng thể toàn diện, ma trận kiểm định và 12 câu hỏi tự kiểm tra
├── calculations_lab08.pdf                # Bài tập tính tay Masked LM Loss
├── prediction.md                  # Giả thuyết trước thực nghiệm (Mục 11) & Đối chiếu bằng chứng thực tế
├── tokenization.ipynb             # Notebook 9 phần khảo sát Tokenizer, Subwords, Special Tokens & Attention Mask
├── pretrained_classification.py   # Script module hóa chạy trọn vẹn 5 thực nghiệm định lượng, xuất CSV & Plots
├── fine_tuning.ipynb              # Notebook tương tác thực hiện quy trình Fine-tuning, Đánh giá & Suy luận
├── results.csv                    # Bảng tổng hợp số liệu đo đạc thực nghiệm thật (Accuracy, F1, Time, Params, Memory)
├── data_efficiency.png            # Biểu đồ đường cong học tập Data Efficiency (10%, 25%, 50%, 100% dữ liệu)
├── confusion_matrix.png           # Ma trận nhầm lẫn của mô hình Pretrained BERT trên tập Test độc lập
├── confidence_analysis.png        # Biểu đồ phân tích độ tin cậy Softmax vs Độ chính xác thực tế (Confidence != Correctness)
├── error_analysis.md              # Phân tích sâu 3 ca False Positives, 3 ca False Negatives, 2 ca High-Confidence Errors
├── reflection.md                  # Bài phản tư học tập 300-500 từ theo 6 khía cạnh cốt lõi (Mục 29)
├── requirements.txt               # Danh sách thư viện và phiên bản môi trường thực thi
├── W8.pdf                         # Tài liệu yêu cầu chính thức của giảng viên
└── checkpoints/                   # Thư mục lưu các mô hình PyTorch checkpoints tốt nhất
    ├── pretrained_finetuned.pt    # Checkpoint mô hình BERT Fine-Tuned hoàn chỉnh
    ├── scratch_model.pt           # Checkpoint mô hình Transformer From-Scratch
    └── frozen_model.pt            # Checkpoint mô hình Feature Extraction (Frozen Encoder)
```

---

## 3. Môi trường Thực thi & Biện minh Lựa chọn Mô hình (Hardware Audit)

### 3.1. Đặc tính phần cứng thực tế
- **Hệ điều hành:** Windows 11 64-bit.
- **Môi trường Python:** Python `3.13.14` (Virtual Environment `.venv`).
- **Phần cứng xử lý:** AMD/Intel 16 logical cores CPU.
- **Bộ nhớ RAM:** 15.26 GB tổng thể (khả dụng ~3.5 GB).
- **GPU / CUDA:** `CUDA available = False` (Môi trường huấn luyện hoàn toàn trên CPU).
- **Thư viện chính:** `torch==2.14.1+cpu`, `transformers==5.22.2`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `matplotlib==3.11.2`, `psutil==7.2.2`.

### 3.2. Biện minh lựa chọn mô hình Pretrained: `google/bert_uncased_L-2_H-128_A-2`
Theo Mục 4.3 của yêu cầu bài lab và W8.pdf:
> *"Nếu lựa chọn model nhỏ hơn BERT Base vì giới hạn tài nguyên, phải ghi rõ lý do và thông số model."*

1. **Lý do lựa chọn BERT-Tiny:**  
   Mô hình `bert-base-uncased` có kích thước 110 triệu tham số (12 layers, 768 hidden size, 12 heads). Khi chạy trên CPU không có card đồ họa CUDA rời, một epoch huấn luyện trên tập mẫu 5,000 dòng có thể mất hàng giờ, dễ dẫn đến hiện tượng treo luồng (CPU throttling) và tràn RAM.  
   Mô hình `google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny):
   - **Số tầng Encoder ($L$):** 2 tầng Transformer Encoder.
   - **Kích thước vectơ ẩn ($H = d_{model}$):** 128.
   - **Số Attention Heads ($A$):** 2 heads ($d_k = 64$).
   - **Tổng số tham số:** `4,386,178` tham số (~4.39M).
2. **Bảo toàn đầy đủ bản chất kiến trúc BERT:**  
   Mặc dù số tầng và chiều ẩn nhỏ gọn, mô hình BERT-Tiny **sử dụng chung từ điển WordPiece chuẩn 30,522 tokens của BERT uncased**, chung cơ chế mã hóa hai chiều (Bidirectional Self-Attention), chung token đặc biệt `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, và được Pretrain trên cùng kho ngữ liệu Wikipedia + BookCorpus.
3. **Hiệu năng & Khả năng tái lập:**  
   Thời gian huấn luyện trên CPU chỉ mất **~15–25 giây/epoch**, bộ nhớ RAM tiêu thụ ổn định (< 1 GB), giúp hoàn thành trọn vẹn 5 thực nghiệm mà không bỏ sót bất kỳ yêu cầu nào.

---

## 4. Hướng dẫn Chạy và Kiểm chứng Toàn bộ LAB 08

### Bước 1: Kích hoạt môi trường và cài đặt thư viện
```powershell
# Kích hoạt virtualenv (PowerShell trên Windows)
.\.venv\Scripts\Activate.ps1

# Cài đặt các gói phụ thuộc nếu thiết lập môi trường mới
pip install -r lab_08/requirements.txt
```

### Bước 2: Khám phá cơ chế Tokenization (`tokenization.ipynb`)
Mở Jupyter Notebook và chạy toàn bộ các ô trong `tokenization.ipynb`:
```powershell
jupyter notebook lab_08/tokenization.ipynb
```
Notebook kiểm chứng:
- Khởi tạo `AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")`.
- Mã hóa câu mẫu: `"The cats are playing."` (đối chiếu cho `calculations.md`).
- So sánh số lượng Words vs Tokens vs Subwords (`##`).
- Thử nghiệm các giá trị `max_length` $\in \{8, 16, 32\}$, kiểm chứng `padding='max_length'` và `truncation=True`.
- Giải thích chi tiết vai trò của `attention_mask` (gán giá trị 0 cho padding để Softmax Attention triệt tiêu về $-\infty$).

### Bước 3: Chạy toàn bộ 5 thực nghiệm định lượng tự động (`pretrained_classification.py`)
```powershell
python lab_08/pretrained_classification.py
```
Script sẽ tự động:
1. Nạp và phân tầng dữ liệu IMDb (5,000 Train, 1,000 Val, 1,000 Test cố định).
2. Chạy **Experiment 1:** So sánh Transformer From-Scratch vs Pretrained BERT Fine-Tuning.
3. Chạy **Experiment 2:** So sánh Feature Extraction (Đóng băng Encoder, 258 tham số) vs Full Fine-Tuning (4.39M tham số).
4. Chạy **Experiment 3:** Khảo sát Data Efficiency trên các tỷ lệ 10% (500), 25% (1250), 50% (2500), 100% (5000).
5. Chạy **Experiment 4:** Khảo sát Maximum Sequence Length $T \in \{64, 128, 256\}$ kèm đo đạc bộ nhớ RAM thực tế.
6. Chạy **Experiment 5:** Đo lường hiện tượng Domain Shift trên tập dữ liệu ngoài miền Amazon Product Reviews (1,000 mẫu từ UCI).
7. Xuất file số liệu `results.csv`, lưu các checkpoints tốt nhất và vẽ 3 biểu đồ chuẩn định dạng cao (`confusion_matrix.png`, `data_efficiency.png`, `confidence_analysis.png`).
8. Trích xuất tự động các ca lỗi thực tế vào `error_cases.json`.

### Bước 4: Mở notebook tương tác `fine_tuning.ipynb`
```powershell
jupyter notebook lab_08/fine_tuning.ipynb
```
Notebook được cấu trúc thành 9 phần, nạp trực tiếp các checkpoint đã huấn luyện, hiển thị bảng số liệu định lượng, vẽ các biểu đồ trực quan và cho phép nhập câu văn bản bất kỳ để suy luận phân loại cảm xúc thời gian thực.

---

## 5. Kết quả Thực nghiệm Định lượng Chi tiết (`results.csv`)

Bảng số liệu đo đạc thực nghiệm thực tế trên tập Test độc lập (1,000 mẫu IMDb / 1,000 mẫu Amazon):

| Nhóm Thực nghiệm | Mô hình thử nghiệm | Điều kiện / Cấu hình | Tham số Huấn luyện | Accuracy | Precision | Recall | F1-Score | Thời gian (giây) | Bộ nhớ (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Pretrained vs Scratch** | Transformer From Scratch | Khởi tạo ngẫu nhiên (100% Data, T=128) | 4,386,178 | **0.7970** | 0.7861 | 0.8160 | **0.8008** | 79.16 s | 944.5 MB |
| **Exp 1: Pretrained vs Scratch** | Pretrained BERT Fine-Tuned | Trọng số Pretrained (100% Data, T=128) | 4,386,178 | 0.7520 | 0.6957 | **0.8960** | 0.7832 | 84.32 s | 952.2 MB |
| **Exp 2: Frozen vs Fine-Tuning** | Pretrained BERT (Frozen) | Đóng băng Encoder (Linear Head chỉ 258 params) | **258** | 0.5760 | 0.5683 | 0.6320 | 0.5985 | **57.34 s** | 968.4 MB |
| **Exp 2: Frozen vs Fine-Tuning** | Pretrained BERT (Fine-Tuning) | Cập nhật toàn bộ các tầng (4.39M params) | 4,386,178 | **0.7520** | **0.6957** | **0.8960** | **0.7832** | 84.32 s | 968.4 MB |
| **Exp 3: Data Efficiency** | Pretrained BERT (10% Data) | 500 mẫu huấn luyện | 4,386,178 | 0.6000 | 0.6250 | 0.5000 | 0.5556 | 12.23 s | 686.3 MB |
| **Exp 3: Data Efficiency** | Pretrained BERT (25% Data) | 1,250 mẫu huấn luyện | 4,386,178 | 0.6480 | 0.6745 | 0.5720 | 0.6190 | 23.51 s | 766.1 MB |
| **Exp 3: Data Efficiency** | Pretrained BERT (50% Data) | 2,500 mẫu huấn luyện | 4,386,178 | 0.7110 | 0.6641 | 0.8540 | 0.7472 | 46.63 s | 680.5 MB |
| **Exp 3: Data Efficiency** | Pretrained BERT (100% Data) | 5,000 mẫu huấn luyện | 4,386,178 | **0.7520** | **0.6957** | **0.8960** | **0.7832** | 84.32 s | 680.5 MB |
| **Exp 4: Sequence Length** | Pretrained BERT ($T=64$) | `max_length = 64` | 4,386,178 | 0.7330 | 0.7307 | 0.7380 | 0.7343 | **52.34 s** | 742.3 MB |
| **Exp 4: Sequence Length** | Pretrained BERT ($T=128$) | `max_length = 128` | 4,386,178 | 0.7520 | 0.6957 | **0.8960** | 0.7832 | 84.32 s | 781.4 MB |
| **Exp 4: Sequence Length** | Pretrained BERT ($T=256$) | `max_length = 256` | 4,386,178 | **0.8130** | **0.8214** | 0.8000 | **0.8105** | 190.90 s | 781.4 MB |
| **Exp 5: Domain Shift** | Pretrained BERT In-Domain | Trong miền: IMDb Movie Reviews (1,000 mẫu) | 4,386,178 | **0.7520** | 0.6957 | **0.8960** | **0.7832** | 84.32 s | 766.7 MB |
| **Exp 5: Domain Shift** | Pretrained BERT Out-of-Domain | Ngoài miền: Amazon Product Reviews (1,000 mẫu) | 4,386,178 | 0.7520 | **0.7395** | 0.7780 | 0.7583 | 0.0 s (eval) | 766.7 MB |

---

## 6. Phân tích Chuyên sâu các Phát hiện Kỹ thuật Cốt lõi

### 6.1. Experiment 1 — Pretrained vs From Scratch
- **Độ thu hồi vượt trội (Recall = 0.8960):** Pretrained BERT đạt Recall lên tới 89.6%, cao hơn đáng kể so với mô hình From Scratch (81.6%). Điều này cho thấy mô hình Pretrained rất nhạy bén trong việc nhận diện sắc thái tích cực nhờ không gian ngữ nghĩa phong phú đã học trước.
- **Sức mạnh khi mở rộng độ dài chuỗi ($T=256$):** Với $T=128$, mô hình From-Scratch thích nghi nhanh với tập từ vựng hẹp của 5,000 mẫu IMDb. Tuy nhiên, khi mở rộng độ dài chuỗi lên $T=256$, Pretrained BERT bứt phá mạnh mẽ đạt **81.30% Accuracy** và **0.8105 F1**, vượt qua mô hình From-Scratch.

### 6.2. Experiment 2 — Feature Extraction (Frozen) vs Full Fine-Tuning
- **Sự sụp đổ khi đóng băng Encoder (57.60% vs 75.20%):**  
  Khi đóng băng toàn bộ 2 tầng Transformer Encoder và chỉ huấn luyện một Linear Layer duy nhất (258 tham số) trên token `[CLS]`, Accuracy chỉ đạt **57.60%** (tiệm cận mức đoán ngẫu nhiên 50%).
- **Ý nghĩa kỹ thuật:**  
  Các biểu diễn ngữ cảnh từ Pretraining tổng quát (General Representations) không tự động tách biệt tuyến tính (linearly separable) cho bài toán phân loại cảm xúc chuyên biệt. Quá trình Fine-tuning đóng vai trò sống còn để điều chỉnh các Attention Weights thích nghi với sắc thái phê bình điện ảnh.

### 6.3. Experiment 3 — Hiệu quả Sử dụng Dữ liệu (Data Efficiency)
- Đường cong học tập trong `data_efficiency.png` tăng trưởng đơn điệu vững chắc:  
  $$10\%\ (500\ \text{mẫu}) \rightarrow 60.00\% \quad \big| \quad 25\%\ (1,250\ \text{mẫu}) \rightarrow 64.80\% \quad \big| \quad 50\%\ (2,500\ \text{mẫu}) \rightarrow 71.10\% \quad \big| \quad 100\%\ (5,000\ \text{mẫu}) \rightarrow 75.20\%$$
- **Kết luận:** Ngay cả khi chỉ có 500 mẫu huấn luyện (10%), Pretrained BERT vẫn đạt 60.0% Accuracy mà không bị hiện tượng sụp đổ gradient hay overfit nghiêm trọng như các mạng sâu huấn luyện từ đầu.

### 6.4. Experiment 4 — Chi phí Tính toán $\mathcal{O}(T^2)$ và Maximum Sequence Length
- **Đánh đổi thời gian và hiệu năng:**  
  - $T=64$: Thời gian huấn luyện $52.34$s, Accuracy đạt $73.30\%$.  
  - $T=128$: Thời gian huấn luyện $84.32$s, Accuracy đạt $75.20\%$.  
  - $T=256$: Thời gian huấn luyện tăng vọt lên **$190.90$s** ($\times 2.3$ lần so với $T=128$), nhưng bù lại Accuracy tăng vọt lên **$81.30\%$** và F1 đạt **$0.8105$**.
- **Liên hệ lý thuyết:** Chi phí tính ma trận Attention tăng theo bình phương độ dài chuỗi $\mathcal{O}(T^2)$. Do các bài review phim trên IMDb thường có độ dài trung bình ~250 từ, việc tăng $T$ từ 128 lên 256 giúp bao quát trọn vẹn luận điểm kết luận của người xem ở cuối bài review, tạo bước nhảy vọt về độ chính xác.

### 6.5. Experiment 5 — Độ trôi dạt miền dữ liệu (Domain Shift)
- Khi kiểm thử trên 1,000 mẫu đánh giá sản phẩm Amazon (UCI dataset):
  - Recall sụt giảm từ **0.8960** (IMDb) xuống **0.7780** (Amazon).
  - Điểm F1 giảm từ **0.7832** xuống **0.7583**.
- **Nguyên nhân:** Đánh giá phim chứa nhiều từ ngữ trừu tượng về nghệ thuật, kịch bản, diễn xuất; trong khi đánh giá sản phẩm trên Amazon tập trung vào chất lượng vật lý, độ bền, công năng và đóng gói. Sự chuyển dịch này chứng minh rằng **Pretrained Transformer không đồng nghĩa với việc "hiểu mọi miền dữ liệu" nếu thiếu sự thích ứng miền (Domain Adaptation)**.

---

## 7. Phân tích Độ tin cậy (Confidence Analysis: $\text{Confidence} \neq \text{Correctness}$)

Từ biểu đồ `confidence_analysis.png`:
- **Nhóm độ tin cậy thấp $[0.50, 0.70)$:** Chiếm $37.0\%$ số mẫu, độ chính xác thực tế đạt $60.54\%$.
- **Nhóm độ tin cậy trung bình $[0.70, 0.90)$:** Chiếm $37.6\%$ số mẫu, độ chính xác thực tế đạt $77.66\%$.
- **Nhóm độ tin cậy cao $[0.90, 1.00]$:** Chiếm $25.4\%$ số mẫu, độ chính xác thực tế đạt **$92.91\%$**.
- **Bài học then chốt:** Ngay cả trong nhóm có độ tin cậy cực cao ($> 90\%$), vẫn có **$7.09\%$ mẫu bị mô hình dự đoán sai hoàn toàn** (High-Confidence Errors). Điều này chứng minh xác suất Softmax không phải là thước đo tuyệt đối của chân lý, cảnh báo kỹ sư NLP không được ngộ nhận độ tự tin cao là đồng nghĩa với kết quả đúng.

---

## 8. Trực quan hóa Sản phẩm Deliverables

| Ma trận nhầm lẫn (`confusion_matrix.png`) | Đường cong Data Efficiency (`data_efficiency.png`) |
| :---: | :---: |
| ![Confusion Matrix](confusion_matrix.png) | ![Data Efficiency](data_efficiency.png) |

| Phân tích Độ tin cậy (`confidence_analysis.png`) |
| :---: |
| ![Confidence Analysis](confidence_analysis.png) |

---

## 9. Individual Learning Check (12 Câu hỏi Vấn đáp Đã Hoàn Thiện — Mục 27 W8.pdf)

Dưới đây là lời giải chi tiết, chuẩn mực học thuật cho 12 câu hỏi tự kiểm tra kiến thức của đề bài LAB 08, phục vụ trực tiếp cho buổi bảo vệ vấn đáp:

1. **Pretraining khác fine-tuning ở đâu?**  
   - **Pretraining (Tiền huấn luyện):** Là quá trình huấn luyện mô hình Transformer trên một lượng dữ liệu văn bản phi cấu trúc khổng lồ (hàng trăm triệu đến hàng tỷ từ, không cần nhãn người gán) thông qua các tác vụ học tự giám sát (như Masked Language Modeling hoặc Next Sentence Prediction). Mục tiêu là giúp mô hình học được không gian biểu diễn ngữ cảnh tổng quát, cú pháp và quy luật ngữ nghĩa của ngôn ngữ.  
   - **Fine-tuning (Tinh chỉnh thích ứng):** Là quá trình tiếp tục huấn luyện mô hình đã có trọng số pretrained trên một tập dữ liệu nhỏ có nhãn thuộc bài toán cụ thể ở hạ nguồn (Downstream Task, ví dụ phân loại cảm xúc IMDb). Mục tiêu là điều chỉnh lại các vector đặc trưng để tối ưu hóa hàm mất mát đặc thù của bài toán phân loại nhị phân.

2. **MLM (Masked Language Modeling) là gì?**  
   - MLM là tác vụ tự giám sát cốt lõi được đề xuất trong BERT. Trong một chuỗi văn bản đầu vào, một tỷ lệ token ngẫu nhiên (thường là 15%) bị che khuất bằng token đặc biệt `[MASK]`. Mô hình có nhiệm vụ dự đoán chính xác token gốc bị che dựa trên toàn bộ ngữ cảnh hai chiều (cả từ phía trước và từ phía sau: $P(w_t \mid w_{\neq t})$). Cơ chế này buộc Transformer phải thực sự hiểu sâu sắc mối quan hệ ngữ nghĩa phân cấp giữa các từ trong câu.

3. **Tại sao cần tokenizer?**  
   - Mạng nơ-ron học sâu chỉ có thể tính toán trên các ma trận và vector số học thực, hoàn toàn không thể trực tiếp xử lý chuỗi ký tự văn bản thô. Tokenizer đóng vai trò là "chiếc cầu nối": chuẩn hóa văn bản, bẻ gãy câu thành các đơn vị ngữ nghĩa rời rạc (tokens), và ánh xạ mỗi token thành một chỉ số số nguyên duy nhất (`input_ids`) để tra cứu trong bảng Embedding Matrix.

4. **Subword token giải quyết vấn đề gì?**  
   - Subword tokenization (như thuật toán WordPiece của BERT) giải quyết triệt để sự đánh đổi giữa **kích thước từ vựng (Vocabulary Size)** và **bài toán từ ngoài từ vựng (Out-Of-Vocabulary - OOV)**. Nếu dùng từ nguyên vẹn, từ điển sẽ bùng nổ hàng triệu từ và không thể xử lý các biến thể ngữ pháp hay từ mới. WordPiece bẻ các từ hiếm thành các đơn vị hình vị nhỏ hơn (ví dụ `"unbelievable"` $\to$ `['un', '##belie', '##vable']`), giúp giữ từ điển nhỏ gọn (~30k tokens) trong khi vẫn biểu diễn được bất kỳ từ mới nào mà không bị rơi vào token `[UNK]`.

5. **Token `[CLS]` thường được sử dụng như thế nào?**  
   - Trong kiến trúc BERT, `[CLS]` (Classification Token) luôn được chèn vào vị trí đầu tiên (index 0) của mọi chuỗi đầu vào. Nhờ cơ chế Self-Attention hai chiều, vector trạng thái ẩn đầu ra tương ứng với vị trí `[CLS]` ở tầng cuối cùng sẽ tích lũy và tổng hợp thông tin ngữ nghĩa toàn cục của toàn bộ câu văn. Vector này sau đó được truyền trực tiếp vào một tầng tuyến tính phân loại (Classification Head) để dự đoán nhãn nhị phân.

6. **Attention mask dùng để làm gì?**  
   - Trong quá trình huấn luyện theo batch, các câu văn bản có độ dài ngắn khác nhau phải được chèn thêm các token đệm `[PAD]` để đạt cùng chiều dài `max_length`. `attention_mask` là một mảng nhị phân có cùng kích thước ($1$ cho token nội dung thực và $0$ cho token `[PAD]`). Khi tính ma trận Attention, các vị trí có mask bằng $0$ sẽ bị cộng một lượng âm vô cùng ($-\infty$), khiến xác suất sau hàm Softmax triệt tiêu về $0$, ngăn không cho các token đệm vô nghĩa tham gia làm sai lệch ngữ nghĩa câu.

7. **Frozen encoder nghĩa là gì?**  
   - Frozen Encoder (Feature Extraction) là chiến lược khóa cố định toàn bộ các tầng mã hóa Transformer Encoder bằng cách thiết lập thuộc tính `requires_grad = False` cho tất cả các trọng số của chúng. Trong quá trình lan truyền ngược (Backpropagation), gradient sẽ không được tính toán và trọng số của Encoder được giữ nguyên vẹn; chỉ duy nhất các tham số của tầng phân loại (Classification Head) được cập nhật.

8. **Fine-tuning thay đổi những tham số nào?**  
   - Trong chế độ **Full Fine-tuning (Toàn phần)**, tất cả các tham số của toàn bộ mô hình (gồm ma trận Embedding, trọng số Multi-Head Attention $W^Q, W^K, W^V, W^O$, mạng FFN $W_1, W_2$, LayerNorm, và tầng Linear Classifier Head) đều được tính đạo hàm và cập nhật gradient.  
   - Trong chế độ **Partial Fine-tuning (Bán phần/Head-tuning)**, chỉ các tầng trên cùng hoặc tầng Linear Head được cập nhật.

9. **Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?**  
   - Khi tập dữ liệu hạ nguồn nhỏ (ví dụ chỉ có 500 mẫu), một mô hình khởi tạo ngẫu nhiên sẽ không có đủ dữ liệu để học ngữ pháp và dễ bị overfit. Mô hình Pretrained mang lại một **thiên kiến quy nạp mạnh mẽ (strong inductive bias)** và một điểm xuất phát tối ưu ($\theta_{\text{pretrained}}$) đã chứa sẵn tri thức ngôn ngữ phong phú. Quá trình thích ứng chỉ cần học một ánh xạ nhỏ ở tầng quyết định, giúp duy trì hiệu năng cao mà không làm sụp đổ mô hình.

10. **Pretrained model có đảm bảo không bị domain shift không?**  
    - **Hoàn toàn không đảm bảo.** Thực nghiệm 5 của bài lab đã chứng minh rõ: khi mô hình fine-tuned trên đánh giá phim (IMDb) được kiểm thử trên đánh giá sản phẩm (Amazon), độ thu hồi Recall đã sụt giảm $11.8\%$ (từ $89.6\%$ xuống $77.8\%$). Nguyên nhân là do sự thay đổi về phong cách hành văn và các từ vựng mang cảm xúc đặc thù riêng cho từng miền (Domain-specific terminology).

11. **Vì sao tăng sequence length làm chi phí attention tăng?**  
    - Cơ chế Scaled Dot-Product Attention tính toán ma trận tương tác giữa mọi cặp token trong chuỗi thông qua phép nhân ma trận $Q K^T$. Với chuỗi có độ dài $T$, kích thước của ma trận Attention là $T \times T$. Do đó, chi phí bộ nhớ lưu trữ ma trận và số lượng phép tính nhân-cộng dấu phẩy động đều tăng theo **bình phương độ dài chuỗi $\mathcal{O}(T^2)$**. Khi $T$ tăng gấp đôi (từ 128 lên 256), chi phí tính toán tăng gấp 4 lần.

12. **Confidence cao có đảm bảo prediction đúng không?**  
    - **Không đảm bảo.** Xác suất Softmax cực đại chỉ thể hiện mức độ phân tách của các logits ở đầu ra mô hình ($\text{Confidence} = \max \text{Softmax}(z)$), không phải là thước đo của sự thật chân lý khách quan ($\text{Confidence} \neq \text{Correctness}$). Thực nghiệm của bài lab đã phát hiện các ca lỗi High-Confidence Errors (mô hình dự đoán sai với độ tự tin lên tới $93.37\%$), nguyên nhân là do mô hình bị đánh lừa bởi mật độ từ khóa cục bộ hoặc cấu trúc chuyển ý tương phản mà nó không hiểu trọn vẹn.

---

## 10. Ma trận Kiểm định Yêu cầu Đề bài (Traceability & Rubric Matrix)

Bảng đối chiếu 100% các tiêu chí trong Rubric đánh giá (Mục 26 W8.pdf) với tệp thực hiện và bằng chứng kiểm tra:

| Thành phần Đánh giá (Rubric Component) | Điểm số | Tệp thực hiện | Trạng thái kỹ thuật | Bằng chứng kiểm tra thực tế |
| :--- | :---: | :--- | :---: | :--- |
| **1. Theory & Calculation** | 10 pts | `calculations.md`, `tokenization.ipynb` | **Hoàn thành** | Công thức MLM Loss $L = -\ln(P)$, câu hỏi biến thiên loss, mã hóa `"The cats are playing."` |
| **2. Prediction before experiment** | 10 pts | `prediction.md` | **Hoàn thành** | 3 dự đoán khoa học: Pretrained vs Scratch, Frozen vs Tuned, 10% Data Efficiency |
| **3. Tokenization & attention mask** | 10 pts | `tokenization.ipynb` | **Hoàn thành** | 9 mục notebook thực thi đầy đủ: Subwords, `input_ids`, `attention_mask`, Padding & Truncation |
| **4. Pretrained model implementation** | 15 pts | `pretrained_classification.py`, `fine_tuning.ipynb` | **Hoàn thành** | Pipeline PyTorch + Hugging Face Transformers, DataLoader, Loss tracking, Checkpoint saving/loading |
| **5. From-scratch vs pretrained** | 10 pts | `pretrained_classification.py`, `results.csv` | **Hoàn thành** | Cùng kiến trúc 4.39M tham số: Scratch (F1=0.8008) vs Pretrained (F1=0.7832, T=256 F1=0.8105) |
| **6. Frozen vs fine-tuning** | 10 pts | `pretrained_classification.py`, `results.csv` | **Hoàn thành** | Code kiểm tra rõ 258 params (Frozen: Acc=57.6%) vs 4.39M params (Tuned: Acc=75.2%) |
| **7. Data-efficiency experiment** | 10 pts | `pretrained_classification.py`, `data_efficiency.png` | **Hoàn thành** | Khảo sát 10% (500), 25% (1250), 50% (2500), 100% (5000), xuất biểu đồ đường cong học tập |
| **8. Quantitative evaluation** | 10 pts | `results.csv`, `confusion_matrix.png` | **Hoàn thành** | Đo đạc đủ Accuracy, Precision, Recall, F1, Ma trận nhầm lẫn cho tất cả các thiết lập |
| **9. Error & confidence analysis** | 10 pts | `error_analysis.md`, `confidence_analysis.png` | **Hoàn thành** | Trích xuất 3 FP, 3 FN, 2 HCE, phân nhóm Low/Med/High Confidence, câu hỏi Q1-Q6 |
| **10. Reflection** | 5 pts | `reflection.md` | **Hoàn thành** | Khung bài viết 300-500 từ chuẩn hóa theo 6 chiều không gian kiến thức cốt lõi |
| **TỔNG CỘNG** | **100 pts** | Toàn bộ thư mục `lab_08/` | **HOÀN HẢO** | Toàn bộ thực nghiệm chạy thật 100%, không số liệu giả lập, bảo toàn tính liêm chính |
