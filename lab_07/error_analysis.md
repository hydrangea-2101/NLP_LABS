# LAB 07: PHÂN TÍCH LỖI THỰC NGHIỆM (ERROR ANALYSIS)

---

## 1. Bảng tổng hợp các trường hợp phân tích

| STT | Mẫu Test | Nhãn thật (True) | Dự đoán (Pred) | Xác suất Positive | Kết quả | Phân loại hiện tượng / Lỗi |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Sample #0 | Negative (0) | Negative (0) | 0.0046 | Đúng | Negation rõ ràng & Từ vựng cảm xúc tiêu cực nhất quán |
| 2 | Sample #2 | Negative (0) | Negative (0) | 0.2888 | Đúng | Domain-specific terms, Rare words & Phê bình châm biếm |
| 3 | Sample #4 | Positive (1) | Positive (1) | 0.9913 | Đúng | Cảm xúc tích cực trực diện, từ ngữ ca ngợi áp đảo trên cả 4 heads |
| 4 | Sample #1 | Negative (0) | Positive (1) | 0.5606 | **SAI** | Contrastive comparison, Long sequence & Truncation (Cắt mất câu kết chê phim) |
| 5 | Sample #3 | Negative (0) | Positive (1) | 0.9724 | **SAI** | Label Noise / Ambiguous sentiment (Nội dung review khen nhưng nhãn gốc Negative) |
| 6 | Sample #11| Positive (1) | Negative (0) | 0.1904 | **SAI** | Negation & Contrastive clause (Mệnh đề phủ định vế đầu `"don't even like... but"`) |

---

## 2. Phân tích chi tiết 3 trường hợp dự đoán đúng (Correct Cases)

### Trường hợp 1: Sample #0 — Dự đoán đúng nhãn Negative
- **Văn bản gốc (trích đoạn):**
  > *"I didn't like Underdog! I mean it was really unnecessary and needed a big face lift and then it would have been maybe passable but for the main part i didn't like it... Then it was really predictable which makes you feel tired and mad. The humor was really what made me mad it was just to unfunny and somewhat pathetic..."*
- **Nhãn thật:** Negative | **Dự đoán:** Negative (Xác suất Positive: `0.0046`)
- **Phân loại hiện tượng:** Direct Negation & Consistent Negative Tone (`negation`, `long sequence`).
- **Giải thích dựa trên bằng chứng và Attention:**
  - Văn bản lặp lại xuyên suốt các cấu trúc phủ định và tính từ chê bai mạnh: `"didn't like"`, `"unnecessary"`, `"predictable"`, `"tired"`, `"mad"`, `"unfunny"`, `"pathetic"`.
  - Kiểm tra trọng số trên 4 Attention Heads của Transformer: các token nhận sự chú ý cao nhất toàn câu là `unfunny (0.119)`, `predictable (0.035)`, `unnecessary (0.028)`. Cụ thể, **Head 3** tập trung rất mạnh vào `unfunny (0.359)` trong khi **Head 4** tập trung vào `predictable (0.073)`.
  - Sự kết hợp đa chiều từ 4 heads cùng tầng biến đổi phi tuyến FFN giúp mô hình dự đoán chính xác tuyệt đối nhãn Negative với xác suất Positive chỉ `0.0046` (tự tin hơn nhiều so với mức `0.1344` của Self-Attention đơn lẻ ở LAB 06).

---

### Trường hợp 2: Sample #2 — Dự đoán đúng nhãn Negative
- **Văn bản gốc (trích đoạn):**
  > *"Basically what we have here is little more than a remake of the hilarious 1970's classic kitsch horror 'Death Line' which ironically was like this cobblers, also partly filmed at the disused Aldwych underground station... originality is not this film's strong point. Indeed strong points are sadly lacking. The gore ranges from the poor to the unnecessarily over gory whilst the sub-Gollum nutter is never really fully explained..."*
- **Nhãn thật:** Negative | **Dự đoán:** Negative (Xác suất Positive: `0.2888`)
- **Phân loại hiện tượng:** Domain-specific Terms, Rare Words & Irony (`domain-specific term`, `rare word`, `contrast`, `truncation`).
- **Giải thích dựa trên bằng chứng và Attention:**
  - Bài viết chứa nhiều từ hiếm/đặc thù văn hóa Anh (`"kitsch"`, `"cobblers"`, `"Aldwych"`, `"sub-Gollum"`, `"nutter"`) và khen bộ phim gốc năm 1970 (`"hilarious"`, `"classic"`), nhưng đánh giá bộ phim hiện tại rất thấp (`"little more than"`, `"not this film's strong point"`, `"sadly lacking"`, `"poor"`).
  - Trong ma trận Multi-Head Attention, mặc dù **Head 4** chú ý vào tính từ `hilarious (0.085)`, nhưng **Head 2** và **Head 3** lại tập trung ổn định vào `poor (0.053 / 0.070)` và `little (0.037 / 0.057)`, còn **Head 1** bắt cụm `fully (0.096)` và `explained (0.051)`. Nhờ khả năng theo dõi nhiều vị trí đồng thời của Multi-Head Attention, tín hiệu tiêu cực giữ được ưu thế và kéo xác suất Positive xuống `0.2888` ($< 0.5$).

---

### Trường hợp 3: Sample #4 — Dự đoán đúng nhãn Positive
- **Văn bản gốc (trích đoạn):**
  > *"I loved this film, Independent film-making at its best. The cinematography, pacing, rhythm, and acting were perfectly in sync. Fred Carpenters best work to date! The movie is well written with lots of plot twists that take you to a great ending... Great performances from all the actors... the is a wonderful film..."*
- **Nhãn thật:** Positive | **Dự đoán:** Positive (Xác suất Positive: `0.9913`)
- **Phân loại hiện tượng:** Strong Positive Praise & Multi-Head Synergy.
- **Giải thích dựa trên bằng chứng và Attention:**
  - Các token quan trọng nhất xuất hiện rõ nét trong phân phối Attention tổng hợp: `wonderful (0.067)`, `great (0.050)`, `best (0.043)`, `twists (0.022)`.
  - Mỗi head thể hiện sự phân công rõ rệt: **Head 1** gom toàn bộ các vị trí xuất hiện từ `great (0.176, 0.100, 0.098)`, **Head 2** chú ý vào danh từ điện ảnh `twists (0.053)` và `beauty (0.024)`, **Head 3** tập trung vào `wonderful (0.185)` và `loved (0.031)`, còn **Head 4** tập trung vào `best (0.151)`. Kết quả là vector sau Mean Pooling mang sắc thái tích cực cực mạnh (`Prob = 0.9913`).

---

## 3. Phân tích chi tiết 3 trường hợp dự đoán sai (Incorrect Cases)

### Trường hợp 4: Sample #1 — Nhãn thật Negative nhưng mô hình đoán Positive (Prob: 0.5606)
- **Văn bản gốc (trích đoạn):**
  > *"If Fassbinder has made a worse film, I sure don't want to see it! Anyone who complains that his films are too talky and claustrophobic should be forced to view this, to learn to appreciate the more spare style he opted for in **excellent films** like 'The Bitter Tears Of Petra von Kant'... [Cắt tại token 128] ... These are the only moments that display any real human emotion, and prove that at the core of **this horrible film**, there was an excellent film struggling to free itself. **What a waste.**"*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1)
- **Phân loại loại lỗi:** **Contrastive Comparison, Long Sequence & Truncation**.
- **Giải thích dựa trên giả thuyết Attention (Hypothesis):**
  - **Giả thuyết:** Bài đánh giá dài 171 tokens. Khi cắt chuỗi ở `max_length = 128`, toàn bộ 43 tokens cuối chứa kết luận phê phán trực diện nhất (*"at the core of this horrible film... What a waste"*) đã bị cắt bỏ hoàn toàn (`truncation`). Trong 128 tokens đầu còn lại, người viết lại dùng cấu trúc giả định mỉa mai và ca ngợi một bộ phim khác (`"learn to appreciate... in excellent films"`).
  - **Kiểm tra các token trong Attention Pattern:** Các token nhận trọng số cao nhất là `worse (0.070)`, `looks (0.038)`, `excellent (0.035)`, `appreciate (0.033)`. Trong đó **Head 2** chú ý vào `worse (0.146)`, nhưng **Head 1** tập trung vào `excellent (0.113)` và **Head 4** tập trung vào `appreciate (0.071)`.
  - **Giới hạn của cách giải thích (`Attention != explanation`):** Mặc dù từ tiêu cực `"worse"` nhận được trọng số Attention trung bình cao nhất toàn câu (`0.070`), mô hình **vẫn dự đoán nhãn Positive (`0.5606`)**. Điều này chứng minh không thể đồng nhất "Attention weight cao = lý do mô hình ra quyết định", bởi biểu diễn cuối cùng còn phụ thuộc vào nhánh truyền thẳng Residual ($x + F(x)$), phép biến đổi phi tuyến FFN và trung bình cộng Mean Pooling trên toàn bộ 128 vị trí.

---

### Trường hợp 5: Sample #3 — Nhãn thật Negative nhưng mô hình đoán Positive (Prob: 0.9724)
- **Văn bản gốc (trích đoạn):**
  > *"'Ernest Saves Christmas' is comedian Ernest's Christmas special film... This film is probably one of Ernest's **best films**. It's **funny**, and it has some **important lessons to learn** regarding friendship and family and not being afraid to show love. It's not the best Christmas film around, **but it is a good one** to get you in the Christmas spirit. It's a little bit sad that the film is a little dated now."*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1)
- **Phân loại loại lỗi:** **Label Noise / Dataset Ambiguity & Truncation**.
- **Giải thích dựa trên giả thuyết và dữ liệu:**
  - Nội dung văn bản trong 128 tokens đầu hoàn toàn mang sắc thái khen ngợi tích cực: khẳng định đây là một trong những bộ phim hay nhất của Ernest (`"best films"`), hài hước (`"funny"`), ý nghĩa (`"important lessons"`) và đáng xem dịp Giáng sinh (`"a good one"`). Câu chê nhẹ duy nhất ở cuối bài (*"a little bit sad that the film is a little dated now"*) lại nằm ở phần đuôi bị cắt bớt ngoài token 128.
  - **Kiểm tra các token trong Attention Pattern:** **Head 4** tập trung mạnh vào hai lần xuất hiện của từ `best (0.083, 0.081)`, còn **Head 2** và **Head 3** chú ý vào chủ đề phim `saves (0.107)` và `spirit (0.044)`.
  - Mô hình Transformer đã phản ánh đúng ngữ nghĩa bề mặt tích cực của văn bản, nhưng bị ghi nhận là lỗi do nhiễu nhãn (Label Noise) của tập dữ liệu gốc IMDb.

---

### Trường hợp 6: Sample #11 — Nhãn thật Positive nhưng mô hình đoán Negative (Prob: 0.1904)
- **Văn bản gốc (trích đoạn):**
  > *"**I don't even like** watching those late night talk shows, **but** I found this one really interesting. I imagine it's probably close to the truth---it 'feels' like an honest account, if that means anything. Kinda feel for the people somewhat when you watch it. A nice movie for a Saturday night."*
- **Nhãn thật:** Positive (1) | **Dự đoán:** Negative (0)
- **Phân loại loại lỗi:** **Negation, Contrastive Concession & Ambiguous Hedging**.
- **Giải thích dựa trên giả thuyết Attention (Hypothesis):**
  - **Giả thuyết:** Câu văn mở đầu bằng mệnh đề phủ định mạnh (`"I don't even like..."`), sau đó chuyển ý bằng liên từ `"but"` sang khen bộ phim theo giọng điệu khá dè dặt, chứa nhiều từ làm suy giảm mức độ khẳng định (`"imagine"`, `"probably"`, `"if that means anything"`, `"kinda"`, `"somewhat"`).
  - **Kiểm tra các token trong Attention Pattern:** Tổng thể trên 4 heads, từ tích cực `nice (0.070)` thực chất nhận được trọng số Attention cao nhất (đặc biệt ở **Head 3** với `0.166`), theo sau là `even (0.059)`, `imagine (0.055)` và `like (0.037)` (ở **Head 2** với `0.101` và `0.099`).
  - **Phân biệt Model Behavior và Human Interpretation:** Nếu con người chỉ nhìn vào biểu đồ Attention và thấy từ `"nice"` đứng đầu bảng trọng số (`0.070`), ta sẽ lầm tưởng mô hình hiểu đây là câu khen. Nhưng thực tế mô hình lại dự đoán **Negative (`Prob = 0.1904`)** vì qua đường truyền tắt **Residual Connection** và **Mean Pooling**, tổng đóng góp của các từ phủ định (`"don't"`, `"even"`) cùng loạt từ hoài nghi (`"anything"`, `"somewhat"`) đã áp đảo từ `"nice"` đơn lẻ ở cuối câu.

---

## 4. Tổng kết bài học từ phân tích lỗi

1. **Phân biệt rạch ròi giữa Model Behavior và Human Interpretation (`Attention != explanation`):** Phân phối trọng số Attention chỉ cho biết các heads đang trộn thông tin giữa những vị trí nào ở tầng hiện tại, hoàn toàn không phải là lời giải thích nhân quả một chiều cho nhãn phân loại cuối cùng (vốn còn đi qua Residual Connection, FFN, Mean Pooling và Linear Classifier).
2. **Tác động kép của Sequence Length và Truncation:** Nhiều bài đánh giá phim trên IMDb đặt câu kết luận cảm xúc ở cuối đoạn văn dài hơn 128 từ (như Sample #1). Khi giới hạn `max_length = 128` để tiết kiệm chi phí $\mathcal{O}(T^2)$, việc cắt mất phần kết luận sẽ làm sai lệch ngữ cảnh toàn bài.
3. **Giới hạn của Transformer huấn luyện từ đầu (From Scratch):** Dù vượt trội hơn hẳn RNN/LSTM và Self-Attention đơn tầng của LAB 06, một khối Transformer học từ đầu trên tập dữ liệu nhỏ vẫn gặp khó khăn trước các cấu trúc nhượng bộ phức tạp (`"I don't even like X, but..."`) và hiện tượng mỉa mai — tạo động lực trực tiếp để chuyển sang **Pretrained Transformer / BERT** ở LAB 08.
