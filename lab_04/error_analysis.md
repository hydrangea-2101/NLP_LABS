# Error Analysis — IMDB Sentiment Classification

## 1. Mục tiêu

Phân tích một số review được mô hình Logistic Regression dự đoán đúng và dự đoán sai trên tập Validation. Mục đích là quan sát nội dung review, so sánh nhãn thật với nhãn dự đoán và xác định những vấn đề cần kiểm tra thêm.

## 2. Kết quả tổng quan

- Tổng số mẫu Validation: 7.500
- Số mẫu dự đoán đúng: 6.677
- Số mẫu dự đoán sai: 823
- Accuracy: 89,03%

## 3. Phân tích các mẫu dự đoán sai

### Mẫu 1: Review về một bộ phim có cảnh không hay

- **Nhãn thật:** `positive`
- **Nhãn dự đoán:** `negative`
- **Quan sát:** Người viết nói mình thích bộ phim và câu chuyện, nhưng cũng chê một cảnh phim.
- **Nhận xét:** Review có cả lời khen và lời chê, trong khi mô hình dự đoán cảm xúc tiêu cực.

### Mẫu 2: Review phim hành động đại học Ấn Độ

- **Nhãn thật:** `positive`
- **Nhãn dự đoán:** `negative`
- **Quan sát:** Review có câu “If you want to have a great time then this is THE movie to watch”. Người viết còn kể về những tình tiết phi lý của phim bằng giọng hài hước.
- **Nhận xét:** Dù nội dung phim được kể có vẻ phi lý, cách diễn đạt của người viết vẫn thể hiện sự thích thú đối với bộ phim.

### Mẫu 3: Review phim về lính cứu hỏa và cứu hộ

- **Nhãn thật:** `negative`
- **Nhãn dự đoán:** `positive`
- **Quan sát:** Review sử dụng từ “heroes” để nói về những người lính cứu hỏa và nhân viên cứu hộ, nhưng cũng gọi tác phẩm là “vapid play”, mang nghĩa vở kịch nhạt nhẽo.
- **Nhận xét:** Từ ngữ tích cực trong review không nhất thiết thể hiện thái độ tích cực đối với bộ phim. Cần xem từ ngữ đang nói về đối tượng nào và đặt chúng trong ngữ cảnh.

## 4. Phân tích các mẫu dự đoán đúng

### Mẫu 1: Review phim Halloween

- **Nhãn thật:** `positive`
- **Nhãn dự đoán:** `positive`
- **Quan sát:** Người viết nhận xét bộ phim hay hơn mong đợi, có nhiều hành động, hồi hộp và bất ngờ; cuối review chấm phim 9/10.
- **Nhận xét:** Điểm số 9/10 là bằng chứng rõ ràng cho thấy người viết đánh giá tích cực về bộ phim.

### Mẫu 2: Review phim Vampire Assassin

- **Nhãn thật:** `negative`
- **Nhãn dự đoán:** `negative`
- **Quan sát:** Người viết chê bộ phim rất tệ nhưng vẫn cho rằng phim tệ đến mức buồn cười và có thể xem để giải trí.
- **Nhận xét:** Một review tiêu cực vẫn có thể chứa những từ ngữ thể hiện sự thích thú. Cần phân biệt việc người viết thấy một bộ phim buồn cười với việc họ đánh giá chất lượng bộ phim tích cực.

### Mẫu 3: Review phim Joe Don Baker

- **Nhãn thật:** `negative`
- **Nhãn dự đoán:** `negative`
- **Quan sát:** Review chỉ trích mạnh bộ phim.
- **Nhận xét:** Trong mẫu này, nội dung đánh giá tiêu cực phù hợp với nhãn thật và nhãn mô hình dự đoán.

## 5. So sánh các mẫu đúng và sai

Qua những mẫu đã xem, em nhận thấy các review dự đoán đúng thường có cảm xúc tổng thể tương đối rõ ràng. Chẳng hạn, review Halloween thể hiện sự hài lòng qua lời khen và điểm số cao, còn review Joe Don Baker có nội dung chỉ trích mạnh.

Ngược lại, một số review dự đoán sai có cả tín hiệu tích cực và tiêu cực. Ví dụ, người viết có thể thích câu chuyện nhưng chê một cảnh phim, hoặc sử dụng lời khen và giọng hài hước khi nói về một bộ phim có nhiều tình tiết phi lý.

Một điểm đáng chú ý khác là từ ngữ tích cực không phải lúc nào cũng nói về bộ phim. Trong review về lính cứu hỏa, từ “heroes” dùng để mô tả những người lính cứu hỏa và nhân viên cứu hộ, chứ không phải để khen tác phẩm.

## 6. Giả thuyết và hạn chế

Một giả thuyết cần kiểm tra thêm là mô hình có thể gặp khó khăn khi tổng hợp những tín hiệu cảm xúc trái chiều hoặc khi từ ngữ mang sắc thái tích cực xuất hiện trong ngữ cảnh tiêu cực.

Tuy nhiên, những mẫu được xem thủ công chưa đủ để khẳng định đây là nguyên nhân chung của các lỗi dự đoán. Phân tích hiện tại chỉ phản ánh những ví dụ đã quan sát, không đại diện cho toàn bộ 823 mẫu dự đoán sai.

## 7. Kết luận

Error Analysis giúp nhận ra rằng việc phân loại cảm xúc không chỉ phụ thuộc vào những từ riêng lẻ. Ngữ cảnh, đối tượng được nhắc đến và đánh giá tổng thể của người viết đều cần được xem xét.

Qua các ví dụ, em nhận thấy một số review có cảm xúc rõ ràng được dự đoán đúng, trong khi những review có tín hiệu trái chiều có thể bị phân loại sai. Đây là những quan sát ban đầu để định hướng việc kiểm tra mô hình sâu hơn, chứ chưa phải bằng chứng xác định nguyên nhân của từng lỗi.