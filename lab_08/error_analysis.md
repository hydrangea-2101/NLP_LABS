# LAB 08: PHÂN TÍCH LỖI THỰC NGHIỆM (ERROR ANALYSIS)

---

## 1. Bảng tổng hợp các trường hợp phân tích

Dưới đây là 8 ca lỗi điển hình được trích xuất tự động 100% từ tập Test độc lập (1,000 mẫu) của mô hình Pretrained BERT Fine-Tuned (`google/bert_uncased_L-2_H-128_A-2`), đáp ứng trọn vẹn yêu cầu Mục 20 W8.pdf (tối thiểu 3 False Positives, 3 False Negatives và 2 High-Confidence Errors):

| STT | Mẫu Test | Nhãn thật (True) | Dự đoán (Pred) | Xác suất Positive | Kết quả | Phân loại hiện tượng / Lỗi cốt lõi |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Sample #202 | Negative (0) | Positive (1) | **0.9337** | **SAI (FP)** | Long-range Dependency, Contrastive Conjunction (`but`) & Positive Word Density |
| 2 | Sample #743 | Negative (0) | Positive (1) | **0.9322** | **SAI (FP)** | Ambiguous Sentiment, Polite Praise & Context Truncation (Cắt ở token 128) |
| 3 | Sample #181 | Negative (0) | Positive (1) | **0.9308** | **SAI (FP)** | Inter-textual Comparison, Sarcastic Tone & Object Confusion (Khen phim khác làm nền) |
| 4 | Sample #975 | Positive (1) | Negative (0) | **0.0741** | **SAI (FN)** | Domain-Specific Horror Jargon, Negative Density vs Rhetorical Defense |
| 5 | Sample #314 | Positive (1) | Negative (0) | **0.0774** | **SAI (FN)** | Colloquial Hedging, Double Negation (`not too bad`) & Mixed Rating (7/10) |
| 6 | Sample #63  | Positive (1) | Negative (0) | **0.1005** | **SAI (FN)** | Polemical Discourse, Quoted Praise vs Refutation of Film Critics |
| 7 | Sample #202 | Negative (0) | Positive (1) | **0.9337** | **SAI (HCE)** | High-Confidence Error: Cực đại hóa xác suất Softmax do từ khóa cục bộ |
| 8 | Sample #743 | Negative (0) | Positive (1) | **0.9322** | **SAI (HCE)** | High-Confidence Error: Cắt mất 150 từ kết luận chê phim ở nửa sau chuỗi |

---

## 2. Phân tích chi tiết các trường hợp False Positives (Nhãn thật Negative nhưng đoán Positive)

### Trường hợp 1: Sample #202 — FP-1 (Độ tự tin: 93.37%)
- **Văn bản gốc (trích đoạn):**
  > *"The core message is strong, the cast has given it their best shot, the packaging is excellent, but the screenplay is seriously over-dramatized and every cliche in the book on women's suffering in India has been over-used to the max."*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1) (Xác suất Positive: `0.9337`)
- **Phân loại loại lỗi:** **Long-range Dependency, Contrastive Conjunction & Positive Token Density**.
- **Giải thích dựa trên giả thuyết và bằng chứng (Hypothesis):**
  - **Mô hình bị áp đảo bởi chuỗi từ tích cực dồn dập ở nửa đầu câu:** Tác giả mở đầu bằng lời khen ngợi liên tiếp: `"core message is strong"`, `"given it their best shot"`, `"packaging is excellent"`. Mặc dù liên từ tương phản `"but"` đã đảo ngược hoàn toàn thái độ ở vế sau (`"seriously over-dramatized"`, `"cliche... over-used to the max"`), mô hình BERT-Tiny (chỉ có 2 tầng Transformer Encoder) chưa tích lũy đủ độ sâu phân cấp để ức chế (inhibit) tín hiệu tích cực của mệnh đề đứng trước liên từ `"but"`.
  - Kết quả là vector của token `[CLS]` bị chi phối bởi các từ mang trọng số cao `"strong"`, `"best"`, `"excellent"`, dẫn đến dự đoán sai lệch với độ tự tin cực đoan lên tới **$93.37\%$**.

---

### Trường hợp 2: Sample #743 — FP-2 (Độ tự tin: 93.22%)
- **Văn bản gốc (trích đoạn):**
  > *"Remember a film you seemed to enjoy in the past that doesn't quite meet those same feelings as an adult? That occurred to me when I went back to school..the National Lampoon's Class Reunion. The film has a perfect opportunity for laughs, but surprisingly wanders aimlessly as we see a bizarre collection of characters... The film has a few good gags that work... but the film slowly gets worse each passing minute. The film's true problem is that it really doesn't know where to go... The cast is enthusiastic enough, but the material... just doesn't have the quality to hold any interest."*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1) (Xác suất Positive: `0.9322`)
- **Phân loại loại lỗi:** **Ambiguous Sentiment, Polite Praise & Context Truncation**.
- **Giải thích dựa trên giả thuyết và bằng chứng (Hypothesis):**
  - **Tác động của hiện tượng cắt ngắn văn bản (`truncation`):** Bài đánh giá gốc có độ dài hơn 350 từ. Với cấu hình thử nghiệm chuẩn `max_length = 128`, toàn bộ hơn 200 từ ở nửa sau của bài đánh giá—nơi tác giả đưa ra các kết luận chê bai gay gắt nhất (`"unassured direction"`, `"relics that are forgotten rightfully so"`)—đã bị loại bỏ hoàn toàn.
  - Trong phạm vi 128 tokens đầu tiên còn lại, người viết lại sử dụng nhiều từ ngữ khen ngợi lịch thiệp (`"enjoy"`, `"perfect opportunity"`, `"good gags"`, `"enthusiastic"`). Mô hình chỉ tiếp nhận được thông tin khen ngợi bề mặt này và đưa ra phán quyết nhầm lẫn sang nhãn Positive với xác suất `0.9322`.

---

### Trường hợp 3: Sample #181 — FP-3 (Độ tự tin: 93.08%)
- **Văn bản gốc (trích đoạn):**
  > *"Of course I have disappeared into the movies. The Neil Young concert film 'Heart of Gold.' There have been many great concert films through the years. The best being Martin Scorsese's 'The Last Waltz'... A phenomenal concert and a phenomenal film... But 'The Last Waltz' was made when the Band and Neil were in their thirties and 'Heart of Gold' was filmed last year when Neil is in his sixties and his band looks as if they are in their late nineties... Close your eyes and it is a terrific concert, open them and view Dorian Grey's hidden portrait... Ghastly, ghostlike... But, only in a dark bar."*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1) (Xác suất Positive: `0.9308`)
- **Phân loại loại lỗi:** **Inter-textual Comparison, Object Confusion & Sarcastic Tone**.
- **Giải thích dựa trên giả thuyết và bằng chứng (Hypothesis):**
  - **Nhầm lẫn đối tượng được đánh giá trong văn cảnh so sánh:** Tác giả dành phần lớn nửa đầu bài để nhiệt liệt ca ngợi một tác phẩm kinh điển khác (*"The Last Waltz"* của Martin Scorsese với các từ ngữ tâng bốc dồn dập: `"great"`, `"best"`, `"phenomenal concert"`, `"phenomenal film"`). Mục đích thực sự của tác giả là lấy bộ phim đó làm chuẩn mực để chê bai bộ phim hiện tại (*"Heart of Gold"*) trông già nua, *"ghastly"*, *"ghostlike"*.
  - Cơ chế Self-Attention của BERT không phân biệt được ranh giới thực thể giữa "bộ phim đang được đánh giá" và "bộ phim đem ra so sánh", dẫn đến việc gom toàn bộ các tính từ tích cực vào token `[CLS]` và dự đoán sai hoàn toàn.

---

## 3. Phân tích chi tiết các trường hợp False Negatives (Nhãn thật Positive nhưng đoán Negative)

### Trường hợp 4: Sample #975 — FN-1 (Độ tự tin: 92.59%)
- **Văn bản gốc (trích đoạn):**
  > *"Those of the 'Instant Gratification' era of horror films will no doubt complain about this film's pace and lack of gratuitous effects and body count. The fact is, 'The Empty Acre' is a good a example of how independent horror films should be done. If you avoid the indie racks because you are tired of annoying teens or twenty somethings getting killed by some baddie... 'The Empty Acre' is the movie for you... Riveting, well acted, and technically astute, 'The Empty Acre' is a fantastic little indie that thinking horror fans should love."*
- **Nhãn thật:** Positive (1) | **Dự đoán:** Negative (0) (Xác suất Positive: `0.0741`, Xác suất Negative: `0.9259`)
- **Phân loại loại lỗi:** **Domain-Specific Negative Jargon & Rhetorical Defense**.
- **Giải thích dựa trên giả thuyết và bằng chứng (Hypothesis):**
  - **Mô hình bị kích hoạt bởi mật độ dày đặc của từ vựng tiêu cực thể loại kinh dị:** Tác giả khen ngợi bộ phim độc lập này bằng cách chỉ trích các bộ phim kinh dị thương mại thị trường khác, sử dụng một loạt từ mang cảm xúc tiêu cực: `"complain"`, `"lack of effects"`, `"annoying teens"`, `"killed"`, `"baddie"`, `"decaying"`, `"struggling"`, `"disappearance"`, `"devours"`.
  - Mặc dù đoạn kết luận là lời khen ngợi rất rõ ràng (`"riveting"`, `"well acted"`, `"technically astute"`, `"fantastic little indie"`), tổng lượng trọng số kích hoạt tiêu cực ở phần mở bài đã kéo tụt phân phối xác suất về phía nhãn Negative.

---

### Trường hợp 5: Sample #314 — FN-2 (Độ tự tin: 92.26%)
- **Văn bản gốc (trích đoạn):**
  > *"maybe i need to have my head examined,but i thought this was a pretty good movie.the CG is not too bad.i have seen worse.the look of the creatures... was pretty cool.the action scenes involving the snakes was really good,i thought.there are some lapses in logic at times,and the story doesn't always make sense.but for a creature feature,there are a lot worse... i still say it was pretty good.i also liked the music.all in all,i think Dragon Wars is about a 7/10"*
- **Nhãn thật:** Positive (1) | **Dự đoán:** Negative (0) (Xác suất Positive: `0.0774`, Xác suất Negative: `0.9226`)
- **Phân loại loại lỗi:** **Colloquial Hedging, Double Negation & Mixed Rating**.
- **Giải thích dựa trên giả thuyết và bằng chứng (Hypothesis):**
  - **Cấu trúc văn phong ngập ngừng và phủ định kép gây nhiễu:** Người viết sử dụng nhiều mệnh đề rào đón khiêm tốn: `"head examined"`, `"not too bad"`, `"seen worse"`, `"lapses in logic"`, `"doesn't always make sense"`, `"there are a lot worse"`.
  - Cụm từ phủ định kép `"not too bad"` về mặt ngữ nghĩa tiếng Anh là lời khen tương đối ("khá tốt"), nhưng sự xuất hiện cạnh nhau của token `not` và `bad` cùng các nhận xét về lỗ hổng logic đã khiến bộ mã hóa BERT phân loại nhầm bài viết vào nhãn Negative, bất chấp việc tác giả cho điểm 7/10 ở cuối bài.

---

### Trường hợp 6: Sample #63 — FN-3 (Độ tự tin: 89.95%)
- **Văn bản gốc (trích đoạn):**
  > *"Well, I'm not about to add to the diatribe that's been 'preached' about this film. All I can add is that at NO time does it even pretend to be a 'historic' picture. Unlike 'Braveheart', 'Titanic' or 'The Patriot' all of which have very credible and serious historical amendments... Has everyone taken a sense of humor bypass? All I can add is a paraphrase from the good Captain Macleane - 'It looked fantastic and we have a bloody good laugh'. One last point... I wonder if the same people made the same judgment about Moulin Rouge. I suggest not."*
- **Nhãn thật:** Positive (1) | **Dự đoán:** Negative (0) (Xác suất Positive: `0.1005`, Xác suất Negative: `0.8995`)
- **Phân loại loại lỗi:** **Polemical Discourse, Rhetorical Questions & Quoted Praise**.
- **Giải thích dựa trên giả thuyết và bằng chứng (Hypothesis):**
  - **Văn phong bút chiến chống lại những lời chỉ trích:** Người viết bảo vệ bộ phim trước những lời chê bai cay nghiệt bằng việc sử dụng các từ ngữ công kích người khác: `"diatribe"`, `"preached"`, `"humor bypass"`, `"worst of excuses"`. Lời khen bộ phim thực tế lại được gián tiếp trích dẫn qua một câu thoại: *"'It looked fantastic and we have a bloody good laugh'"*.
  - Mô hình Transformer không nhận diện được ngữ cảnh bảo vệ tác phẩm, chỉ ghi nhận các từ vựng tiêu cực trong cuộc tranh luận và đưa ra kết luận sai.

---

## 4. Phân tích các ca High-Confidence Errors (Tự tin cực cao nhưng đoán sai)

Hai trường hợp sai lệch với độ tự tin lớn nhất toàn bộ tập kiểm thử:
1. **HCE-1 (Sample #202 — Confidence: 93.37% Positive):**  
   - Bản chất: Mô hình bị "đánh lừa" bởi mật độ từ vựng khen ngợi cục bộ ở vế đầu câu. Do hàm Softmax sử dụng số mũ $e^{z_i}$, khi logit của nhãn Positive chỉ cần nhỉnh hơn do 3 từ `"strong"`, `"best"`, `"excellent"`, xác suất dự đoán bị phóng đại lên mức $93.37\%$.
2. **HCE-2 (Sample #743 — Confidence: 93.22% Positive):**  
   - Bản chất: Hiện tượng cắt ngắn chuỗi (`truncation`) ở token 128 làm mất hoàn toàn phần kết luận chê bai ở nửa sau, buộc mô hình phải đưa ra phán quyết tuyệt đối dựa trên phần hồi tưởng tích cực ở nửa đầu.
- **Bài học then chốt:** $\mathbf{\text{Confidence} \neq \text{Correctness}}$. Độ tự tin Softmax phản ánh sự tách biệt của các logit đầu ra, không đồng nghĩa với tính chân lý khách quan của dự đoán.

---

## 5. Trả lời các câu hỏi bắt buộc (Section 28 W8.pdf)

---

### Q1: Model nào hưởng lợi nhiều hơn khi training data ít?
- **Observation:** Trong Thí nghiệm 3 (Data Efficiency), khi tập huấn luyện giảm 10 lần (từ $5,000$ xuống còn $500$ mẫu), Pretrained BERT vẫn duy trì độ chính xác $Accuracy = 60.00\%$ và $F1 = 0.5556$.
- **Evidence:** Mô hình Transformer From-Scratch (với hơn $4.38\text{M}$ tham số ngẫu nhiên) khi học trên 500 mẫu bị sụp đổ gradient và mất khả năng tổng quát hóa do thiếu dữ liệu trầm trọng.
- **Interpretation:** **Mô hình Pretrained Transformer hưởng lợi nhiều hơn vượt trội khi dữ liệu ít.** Tri thức biểu diễn ngữ cảnh được nạp sẵn từ hàng tỷ tokens trong giai đoạn Pretraining cung cấp một điểm xuất phát tối ưu ($\theta_{\text{pretrained}}$), giúp mô hình chỉ cần một lượng nhỏ dữ liệu để căn chỉnh mặt phẳng phân tách phân loại.

---

### Q2: Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
- **Observation:** Ở độ dài chuỗi $T=256$, Pretrained BERT đạt $Accuracy = 81.30\%$ và $F1 = 0.8105$ (vượt qua From-Scratch $79.70\%$ và $0.8008$), đồng thời đạt độ thu hồi vượt bậc ($Recall = 0.8960$ so với $0.8160$).
- **Evidence:** Pretrained BERT hội tụ với hàm mất mát mượt mà hơn và nhận diện chính xác các văn bản mang cấu trúc ngữ pháp phức tạp.
- **Interpretation:** Điều này chứng minh **giá trị chuyển giao tri thức (Transferability of Representations)** của học tự giám sát (Self-supervised learning). Nhiệm vụ Masked Language Modeling giúp mạng Transformer nắm bắt được các đặc trưng phân cấp của ngôn ngữ (ngữ pháp, liên kết từ, cấu trúc câu), mang lại lợi thế thích ứng áp đảo so với việc học từ đầu trên một tập dữ liệu nhỏ.

---

### Q3: Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
- **Observation:** Trong thực nghiệm của LAB 08, Frozen Model chỉ đạt $Accuracy = 57.60\%$ và $F1 = 0.5985$, trong khi Full Fine-Tuning đạt $Accuracy = 75.20\%$ và $F1 = 0.7832$ (chênh lệch tới $+17.60\%$).
- **Evidence:** Hai kết quả này **hoàn toàn không gần bằng nhau**, mà có sự chênh lệch mang tính bản chất.
- **Interpretation:** Nếu trên lý thuyết một mô hình Frozen đạt kết quả gần bằng Fine-tuned, điều đó có nghĩa là các đặc trưng biểu diễn tĩnh (Static Contextual Embeddings) của Pretrained Encoder đã tạo ra các cụm dữ liệu phân tách tuyến tính (linearly separable) hoàn hảo cho bài toán hạ nguồn. Tuy nhiên, sự chênh lệch lớn trong thực tế chứng minh rằng biểu diễn tổng quát của BERT tiền huấn luyện vẫn mang tính đa mục đích; nếu không cập nhật các tầng Attention trên cùng để thích nghi chuyên biệt với sắc thái cảm xúc (Task Adaptation), mô hình không thể phân loại chính xác các nhận xét phim mang tính lưỡng phân.

---

### Q4: Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
- **Observation:** Fine-tuning đạt $75.20\%$ Accuracy, bỏ xa mức $57.60\%$ của Frozen Model tới $17.60$ điểm phần trăm.
- **Evidence:** Toàn bộ $4,386,178$ tham số khi được cập nhật gradient đã tự động tái phân bổ trọng số trong các ma trận chiếu $W^Q, W^K, W^V$ của 2 tầng Multi-Head Attention.
- **Interpretation:** Điều này gợi ý rằng **tập dữ liệu đánh giá phim IMDb có những đặc thù ngữ nghĩa rất riêng biệt** (từ ngữ cảm xúc, cấu trúc phủ định, thành ngữ điện ảnh) mà bài toán Masked Language Modeling tổng quát trên Wikipedia chưa từng được định hướng để tối ưu hóa. Fine-tuning là bước chuyển đổi bắt buộc để "xoay chuyển" không gian tiềm ẩn (latent space), đưa các vector biểu diễn của câu tích cực và tiêu cực dạt về hai cực đối lập nhau trước khi đưa vào Linear Classifier Head.

---

### Q5: Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
- **Observation:** Trong Thí nghiệm 5 (Domain Shift), khi kiểm thử mô hình đã fine-tuned từ IMDb sang tập đánh giá sản phẩm Amazon (UCI), độ thu hồi $Recall$ giảm mạnh từ **$0.8960$ xuống $0.7780$** (sụt giảm $11.8\%$).
- **Evidence:** Cùng là bài toán phân loại nhị phân Positive/Negative, nhưng các câu đánh giá sản phẩm Amazon tập trung vào thuộc tính vật lý, kích thước, độ bền và cổng kết nối.
- **Interpretation:** Nguyên nhân cốt lõi là **sự trôi dạt phân phối từ vựng và phong cách hành văn (Distributional Covariate Shift & Concept Drift)**:  
  1. *Khác biệt phong cách:* IMDb gồm các bài luận dài, văn phong nghệ thuật, nhiều liên từ tương phản và so sánh bóng bẩy; Amazon gồm các câu ngắn gọn, văn nói trực diện về chức năng thiết bị.  
  2. *Đặc thù từ vựng cảm xúc:* Trên IMDb, các từ như `"gripping"`, `"cinematography"`, `"tearjerker"` thể hiện sự khen ngợi; trên Amazon, các từ thể hiện cảm xúc tích cực lại là `"sturdy"`, `"fast shipping"`, `"fits perfectly"`, `"battery life"`. Mô hình fine-tuned trên IMDb bị thiên lệch sang miền điện ảnh và không phản xạ nhạy bén với từ vựng của miền sản phẩm tiêu dùng.

---

### Q6: Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
- **Observation:** Khi tăng $T$ từ $64$ lên $128$, Accuracy tăng từ $73.30\%$ lên $75.20\%$ ($+1.9\%$, thời gian tăng từ $52\text{s} \to 84\text{s}$). Khi tăng từ $128$ lên $256$, Accuracy tăng lên $81.30\%$ ($+6.1\%$, thời gian tăng vọt lên $190.90\text{s}$).
- **Evidence:** Chi phí tính toán của cơ chế Self-Attention tỷ lệ thuận với bình phương chiều dài chuỗi $\mathcal{O}(T^2)$.
- **Interpretation:** **Không nên tiếp tục tăng sequence length lên $T=512$ nếu lợi ích biên bắt đầu bão hòa.**  
  Trong bài toán IMDb, $T=256$ đã bao quát được gần $85\%$ độ dài của các bài đánh giá thông thường, mang lại bước nhảy vọt $81.30\%$. Nếu tiếp tục nâng lên $T=512$, thời gian huấn luyện sẽ tăng gấp 4 lần ($> 800$ giây) và bộ nhớ RAM sẽ tăng mạnh, nhưng phần lớn các token ở vị trí $> 256$ là padding tokens (`[PAD]`) hoặc nội dung lặp lại, không cung cấp thêm tín hiệu phân loại mới. Kỹ sư NLP cần cân nhắc điểm cân bằng Pareto giữa độ chính xác và chi phí tài nguyên phần cứng.
