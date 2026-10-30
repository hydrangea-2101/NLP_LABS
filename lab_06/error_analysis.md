# LAB 06: PHÂN TÍCH LỖI THỰC NGHIỆM (ERROR ANALYSIS)

---

## 1. Bảng tổng hợp các trường hợp phân tích

| STT | Mẫu Test | Nhãn thật (True) | Dự đoán (Pred) | Xác suất Positive | Kết quả | Phân loại hiện tượng / Lỗi |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Sample #0 | Negative (0) | Negative (0) | 0.1344 | Đúng | Phủ định rõ ràng & Từ vựng cảm xúc tiêu cực nhất quán |
| 2 | Sample #2 | Negative (0) | Negative (0) | 0.4440 | Đúng | Phê bình châm biếm nhẹ kết hợp nhiều từ vựng tiêu cực |
| 3 | Sample #4 | Positive (1) | Positive (1) | 0.6724 | Đúng | Cảm xúc tích cực trực diện, từ ngữ ca ngợi áp đảo |
| 4 | Sample #1 | Negative (0) | Positive (1) | 0.7972 | **SAI** | Ambiguous wording & Sarcasm (So sánh với phim xuất sắc khác) |
| 5 | Sample #3 | Negative (0) | Positive (1) | 0.9407 | **SAI** | Label Noise / Ambiguous sentiment (Review thực chất khen) |
| 6 | Sample #11| Positive (1) | Negative (0) | 0.1790 | **SAI** | Negation & Long-range contrastive clause (Vế đầu "don't like") |

---

## 2. Phân tích chi tiết 3 trường hợp dự đoán đúng (Correct Cases)

### Trường hợp 1: Sample #0 — Dự đoán đúng nhãn Negative
- **Văn bản gốc (trích đoạn):**
  > *"I didn't like Underdog! I mean it was really unnecessary and needed a big face lift and then it would have been maybe passable but for the main part i didn't like it... Then it was really predictable which makes you feel tired and mad. The humor was really what made me mad it was just to unfunny and somewhat pathetic..."*
- **Nhãn thật:** Negative | **Dự đoán:** Negative (Xác suất Positive: `0.1344`)
- **Phân loại hiện tượng:** Direct Negative Sentiment & Consistent Tone.
- **Giải thích dựa trên bằng chứng và Attention:**
  - Văn bản có cấu trúc cảm xúc đồng nhất, lặp lại nhiều lần cụm từ tiêu cực mạnh mẽ: `"didn't like"`, `"unnecessary"`, `"predictable"`, `"tired"`, `"mad"`, `"unfunny"`, `"pathetic"`.
  - Cơ chế Self-Attention kết nối các token mang sắc thái chê bai trực tiếp này với nhau. Ngay cả khi cơ chế tính trung bình pooling được áp dụng, vector ngữ cảnh tổng thể vẫn lệch hẳn về phía cực tiêu cực trong không gian embedding.

---

### Trường hợp 2: Sample #2 — Dự đoán đúng nhãn Negative
- **Văn bản gốc (trích đoạn):**
  > *"Basically what we have here is little more than a remake of the hilarious 1970's classic kitsch horror 'Death Line'... originality is not this film's strong point. Indeed strong points are sadly lacking. The gore ranges from the poor to the unnecessarily over gory whilst the sub-Gollum nutter is never really fully explained... out-acted by a small dog and a pack of tame rats..."*
- **Nhãn thật:** Negative | **Dự đoán:** Negative (Xác suất Positive: `0.4440`)
- **Phân loại hiện tượng:** Irony / Mild Sarcasm with Negative Keywords.
- **Giải thích dựa trên bằng chứng và Attention:**
  - Mặc dù tác giả dùng từ `"hilarious"`, `"classic"` khi nhắc đến bộ phim cũ, nhưng phần đánh giá phim hiện tại xuất hiện nhiều từ khóa tiêu cực dày đặc: `"sadly lacking"`, `"poor"`, `"unnecessarily over gory"`, `"out-acted"`.
  - Attention phân tán giữa việc mô tả cốt truyện và các nhận xét tiêu cực, khiến xác suất tích cực xấp xỉ `0.444` (nghiêng nhẹ về phía tiêu cực, vượt qua ngưỡng phân loại 0.5 vừa đủ).

---

### Trường hợp 3: Sample #4 — Dự đoán đúng nhãn Positive
- **Văn bản gốc (trích đoạn):**
  > *"I loved this film, Independent film-making at its best. The cinematography, pacing, rhythm, and acting were perfectly in sync. Fred Carpenters best work to date! The movie is well written with lots of plot twists that take you to a great ending... Great performances from all the actors... You will love it."*
- **Nhãn thật:** Positive | **Dự đoán:** Positive (Xác suất Positive: `0.6724`)
- **Phân loại hiện tượng:** Direct Positive Praise & Strong Semantic Affinity.
- **Giải thích dựa trên bằng chứng và Attention:**
  - Toàn bộ đoạn văn tràn ngập các từ khóa ca ngợi xuất sắc: `"loved"`, `"best"`, `"perfectly in sync"`, `"well written"`, `"great ending"`, `"wonderful"`.
  - Các token này đóng vai trò các Key có độ tương đồng rất cao với các truy vấn ngữ nghĩa tích cực, tạo nên các Attention Weights tập trung cao tại các tính từ khen ngợi, dẫn tới phân loại Positive hoàn toàn chính xác.

---

## 3. Phân tích chi tiết 3 trường hợp dự đoán sai (Incorrect Cases)

### Trường hợp 4: Sample #1 — Nhãn thật Negative nhưng mô hình đoán Positive (Prob: 0.7972)
- **Văn bản gốc (trích đoạn):**
  > *"If Fassbinder has made a worse film, I sure don't want to see it! Anyone who complains that his films are too talky and claustrophobic should be forced to view this, to learn to appreciate the more spare style he opted for in **excellent films** like 'The Bitter Tears Of Petra von Kant'... These are the only moments that display any **real human emotion**, and prove that at the core of this horrible film, there was an **excellent film struggling to free itself**. What a waste."*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1)
- **Phân loại loại lỗi:** **Ambiguous wording, Contrastive Reference & Sarcasm**.
- **Giải thích dựa trên giả thuyết Attention (Hypothesis):**
  - **Giả thuyết:** Người viết dùng kỹ thuật so sánh đối lập: chê bộ phim hiện tại thậm tệ (`"worse film"`, `"horrible film"`, `"What a waste"`) bằng cách ca ngợi các tác phẩm khác của cùng đạo diễn (`"excellent films"`, `"real human emotion"`, `"excellent film struggling to free itself"`).
  - Từ ngữ khen ngợi `"excellent"` xuất hiện lặp lại và có trọng lượng rất lớn trong từ điển sentiment. Lớp Self-Attention đơn tầng (Single-layer Attention không có Positional Encoding sâu và Multi-Head) đã tính tương quan mạnh giữa các từ `"excellent"`, `"real"`, `"emotion"` với ngữ cảnh chung, lấn át các từ chê bai nằm rải rác ở đầu và cuối chuỗi.
  - **Giới hạn của cách giải thích:** Ta không thể khẳng định chắc chắn 100% rằng "chỉ vì Attention nhìn vào từ *excellent* nên mô hình bị lừa". Sự sai lệch này còn đến từ việc bộ phân loại phía sau (Classifier MLP) lấy trung bình pooling trên toàn bộ sequence (Mean Pooling), khiến các biểu diễn từ tích cực áp đảo về tổng đại số.

---

### Trường hợp 5: Sample #3 — Nhãn thật Negative nhưng mô hình đoán Positive (Prob: 0.9407)
- **Văn bản gốc (trích đoạn):**
  > *"'Ernest Saves Christmas' is comedian Ernest's Christmas special film... This film is probably one of Ernest's **best films**. It's **funny**, and it has some **important lessons to learn** regarding friendship and family and not being afraid to show love. It's not the best Christmas film around, but it is a **good one** to get you in the Christmas spirit..."*
- **Nhãn thật:** Negative (0) | **Dự đoán:** Positive (1)
- **Phân loại loại lỗi:** **Label Noise / Dataset Ambiguity**.
- **Giải thích dựa trên giả thuyết và dữ liệu:**
  - Khi đọc trực tiếp nội dung review của con người, đây rõ ràng là một lời khen ngợi tích cực: người xem khẳng định đây là phim hay nhất của Ernest (`"best films"`), hài hước (`"funny"`), mang nhiều bài học quý giá và là một bộ phim hay (`"good one"`).
  - Tuy nhiên, trong tập dữ liệu IMDB Dataset ban đầu, mẫu này bị gắn nhãn nhầm là `negative` (hoặc do người chấm điểm gốc cho 5/10 sao nhưng lời bình lại mang hướng khen).
  - Do đó, **mô hình Self-Attention thực ra đã nắm bắt chính xác ngữ nghĩa bề mặt của văn bản**, nhưng bị tính là một lỗi sai do nhiễu nhãn (Label Noise) của tập dữ liệu kiểm thử.

---

### Trường hợp 6: Sample #11 — Nhãn thật Positive nhưng mô hình đoán Negative (Prob: 0.1790)
- **Văn bản gốc (trích đoạn):**
  > *"**I don't even like** watching those late night talk shows, **but** I found this one really interesting. I imagine it's probably close to the truth---it 'feels' like an honest account, if that means anything. Kinda feel for the people somewhat when you watch it. A nice movie for a Saturday night."*
- **Nhãn thật:** Positive (1) | **Dự đoán:** Negative (0)
- **Phân loại loại lỗi:** **Negation & Long-range Contrastive Clause (Mệnh đề phủ định vế trước)**.
- **Giải thích dựa trên giả thuyết Attention (Hypothesis):**
  - **Giả thuyết:** Câu mở đầu bằng một cụm phủ định sở thích cá nhân: *"I don't even like..."*. Ngay sau đó, tác giả dùng liên từ chuyển ý *"but"* để khẳng định rằng tác phẩm này lại ngoại lệ rất thú vị (*"really interesting... honest account... nice movie"*).
  - Trong mô hình Self-Attention đơn tầng, token `"don't"` và `"like"` tạo ra tương tác trực tiếp rất mạnh. Khi vector ngữ cảnh được tổng hợp, vế phủ định mạnh mẽ ở đầu câu đã kéo toàn bộ phân phối cảm xúc xuống mức tiêu cực (xác suất positive chỉ đạt `0.1790`).
  - Mô hình chưa học được cơ chế trọng số chuyển hướng (discourse shift) mà liên từ phản đề `"but"` mang lại, vốn đòi hỏi cơ chế Multi-Head Attention hoặc kiến trúc sâu hơn (Transformer / BERT) để tách biệt vế phụ và vế chính.

---

## 4. Tổng kết bài học từ phân tích lỗi

1. **Attention không phải là phép màu suy luận:** Mặc dù Attention cho phép các từ cách xa nhau tương tác trực tiếp (vượt qua giới hạn nén cổ chai của RNN/LSTM), nhưng bản thân trọng số Attention chỉ thể hiện mức độ tương đồng trong không gian biểu diễn toán học, không phản ánh sự hiểu biết logic ngôn ngữ.
2. **Hiện tượng bẫy từ vựng cảm xúc (Sentiment Word Trap):** Trong các câu chứa châm biếm (sarcasm) hoặc so sánh với tác phẩm khác, các từ mang cảm xúc cực đoan (như *"excellent"*, *"horrible"*) rất dễ chi phối toàn bộ ma trận Attention.
3. **Cần các thành phần bổ trợ trong Transformer:** Để xử lý triệt để các lỗi trên, mô hình cần:
   - *Positional Encoding* chuẩn xác để phân biệt vị trí trước/sau liên từ *"but"*.
   - *Multi-Head Attention* để một head tập trung vào quan hệ cú pháp (Subject-Verb-Object), một head khác tập trung vào phạm vi phủ định (Negation scope).
