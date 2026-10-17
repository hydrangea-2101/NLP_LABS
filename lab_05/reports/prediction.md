### Prediction 1: TF-IDF vs RNN

**Prediction:** RNN có khả năng khai thác thông tin về thứ tự từ tốt hơn TF-IDF thông thường, nhưng không nhất thiết phân loại cảm xúc tốt hơn.

**Reason:** TF-IDF biểu diễn văn bản dựa trên trọng số của các từ mà không giữ lại thứ tự từ. RNN xử lý các token theo trình tự và cập nhật trạng thái ẩn, nên có khả năng học thông tin về thứ tự. Tuy nhiên, khả năng biểu diễn thông tin tốt hơn ở một khía cạnh không đồng nghĩa với hiệu quả phân loại cao hơn. Cần đánh giá các mô hình trên cùng tập dữ liệu kiểm thử bằng những chỉ số phù hợp.

**Confidence:** Cao.