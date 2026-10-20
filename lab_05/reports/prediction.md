### Prediction 1: TF-IDF vs RNN

**Prediction:** RNN có khả năng khai thác thông tin về thứ tự từ tốt hơn TF-IDF thông thường, nhưng không nhất thiết phân loại cảm xúc tốt hơn.

**Reason:** TF-IDF biểu diễn văn bản dựa trên trọng số của các từ mà không giữ lại thứ tự từ. RNN xử lý các token theo trình tự và cập nhật trạng thái ẩn, nên có khả năng học thông tin về thứ tự. Tuy nhiên, khả năng biểu diễn thông tin tốt hơn ở một khía cạnh không đồng nghĩa với hiệu quả phân loại cao hơn. Cần đánh giá các mô hình trên cùng tập dữ liệu kiểm thử bằng những chỉ số phù hợp.

**Confidence:** Cao.

### Prediction 2: RNN vs LSTM

**Prediction:** LSTM được kỳ vọng xử lý các phụ thuộc dài hạn tốt hơn RNN thông thường, nhưng không nhất thiết đạt độ chính xác phân loại cao hơn trong mọi trường hợp.

**Reason:** RNN cập nhật trạng thái ẩn dựa trên token hiện tại và trạng thái ẩn trước đó. Khi chuỗi dài, mô hình có thể gặp khó khăn trong việc duy trì thông tin từ những token ở đầu chuỗi, đặc biệt do hiện tượng tiêu biến gradient trong quá trình huấn luyện. LSTM bổ sung cell state và các cổng điều khiển việc giữ lại, cập nhật và thể hiện thông tin, nhờ đó có thể duy trì thông tin quan trọng qua nhiều bước thời gian tốt hơn. Tuy nhiên, nếu bài toán không đòi hỏi phụ thuộc dài hạn, RNN vẫn có thể hoạt động tốt. Kết quả thực tế còn phụ thuộc vào dữ liệu, kiến trúc và quá trình huấn luyện.

**Confidence:** Cao.

### Prediction 3:

**Prediction:** Khi tăng `max_length` từ 50 lên 200, thời gian huấn luyện dự kiến sẽ tăng.

**Reason:** Với `max_length = 200`, mô hình phải xử lý chuỗi có tối đa 200 vị trí token thay vì 50. Các mô hình RNN, LSTM và GRU thực hiện tính toán tuần tự qua các bước thời gian, vì vậy chuỗi dài hơn thường làm tăng số phép tính trong quá trình huấn luyện. Tuy nhiên, thời gian thực tế không nhất thiết tăng đúng bốn lần vì còn phụ thuộc vào kiến trúc, phần cứng và cách triển khai. Accuracy cũng không nhất thiết tăng khi độ dài chuỗi tăng.

**Confidence:** Cao.

### Prediction 4:

**Prediction:** Nếu chỉ giữ lại 30 token đầu tiên của mỗi bài đánh giá, hiệu quả phân loại cảm xúc có thể giảm do mô hình mất một phần thông tin quan trọng.

**Reason:** Khi cắt chuỗi tại 30 token, toàn bộ nội dung phía sau vị trí này sẽ bị loại bỏ. Nếu phần bị loại chứa các từ thể hiện cảm xúc, câu phủ định hoặc kết luận của người viết, mô hình có thể không hiểu đầy đủ ý nghĩa của bài đánh giá và dự đoán sai nhãn cảm xúc. Tuy nhiên, mức độ ảnh hưởng phụ thuộc vào vị trí và mức độ quan trọng của thông tin bị loại bỏ, nên accuracy không nhất thiết giảm trong mọi trường hợp.

**Confidence:** Cao.

### Prediction 5:

**Prediction:** LSTM không nhất thiết đạt accuracy cao hơn RNN trong mọi bài toán phân loại cảm xúc.

**Reason:** LSTM có cell state và các cổng điều khiển bộ nhớ, giúp mô hình có khả năng xử lý phụ thuộc dài hạn tốt hơn RNN thông thường. Tuy nhiên, không phải bài toán nào cũng cần ghi nhớ thông tin từ xa. Với những bài toán đơn giản, RNN có thể đã hoạt động tốt. Ngoài ra, LSTM có cấu trúc phức tạp hơn và kết quả phụ thuộc vào dữ liệu, kiến trúc cũng như quá trình huấn luyện. Vì vậy, cần so sánh các mô hình trên cùng tập kiểm thử bằng những chỉ số phù hợp thay vì mặc định LSTM luôn tốt hơn.

**Confidence:** Cao.