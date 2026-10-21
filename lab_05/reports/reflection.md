# Tổng kết và tự đánh giá

## 1. Kiến thức đã thực hành

Qua bài thực hành, em đã tiếp cận các phương pháp biểu diễn văn bản và mô hình phân loại cảm xúc.

Các nội dung chính gồm:

- Biểu diễn văn bản bằng TF-IDF.
- Xây dựng từ điển từ vựng và mã hóa token thành số nguyên.
- Xử lý chuỗi văn bản có độ dài khác nhau.
- Tìm hiểu mô hình RNN, LSTM và GRU.
- Đánh giá mô hình bằng Accuracy, Precision, Recall và F1-score.
- Phân tích kết quả dự đoán và xác định các vấn đề cần kiểm tra.

## 2. So sánh mô hình

Trong lần thực nghiệm hiện tại, TF-IDF kết hợp Logistic Regression đạt Accuracy 85%, cao hơn các mô hình RNN, LSTM và GRU.

Các mô hình học sâu lần lượt đạt Accuracy khoảng 53,13%, 50,50% và 50,50%. Vì mỗi mô hình mới được huấn luyện một epoch, kết quả này chưa đủ để đánh giá đầy đủ tiềm năng của các kiến trúc.

Qua đó, em nhận thấy cần đánh giá mô hình dựa trên cả kết quả thực nghiệm và cách thiết lập thí nghiệm, thay vì chỉ dựa vào tên hoặc độ phức tạp của mô hình.

## 3. Kiểm chứng phép tính của Simple RNN Cell

Trong file `rnn_cell.py`, em đã cài đặt phép tính của một ô Simple RNN bằng NumPy và đối chiếu kết quả với `torch.nn.RNNCell`.

Công thức trạng thái ẩn là:

\[
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
\]

Trong đó:

- \(x_t\): đầu vào tại thời điểm hiện tại.
- \(h_{t-1}\): trạng thái ẩn của bước trước.
- \(W_{xh}\): ma trận trọng số từ đầu vào đến trạng thái ẩn.
- \(W_{hh}\): ma trận trọng số trạng thái ẩn.
- \(b_h\): vector độ lệch.
- \(h_t\): trạng thái ẩn mới.

Kết quả kiểm chứng ghi nhận sai khác tuyệt đối lớn nhất giữa hai cách tính là khoảng \(3 \times 10^{-8}\), nhỏ hơn ngưỡng \(10^{-5}\).

Điều này xác nhận rằng phép tính được cài đặt trong bài kiểm chứng khớp với kết quả tham chiếu của PyTorch trong trường hợp thử nghiệm.

## 4. Những hạn chế hiện tại

- Các mô hình RNN, LSTM và GRU mới được huấn luyện một epoch.
- Thí nghiệm so sánh độ dài chuỗi chưa được chạy trong lần thực nghiệm hiện tại.
- Hiệu suất của các mô hình học sâu còn thấp và cần được kiểm tra thêm.
- Chưa có đủ bằng chứng để xác định nguyên nhân chính gây ra các dự đoán sai.

## 5. Hướng phát triển tiếp theo

1. Kiểm tra lại toàn bộ quy trình tiền xử lý và mã hóa nhãn.
2. Huấn luyện mô hình trong nhiều epoch hơn.
3. Theo dõi Loss và các chỉ số trên tập Validation.
4. Thực hiện thí nghiệm với các độ dài chuỗi khác nhau.
5. Phân tích các trường hợp dự đoán sai và ghi nhận kết quả sau mỗi lần cải thiện.

## 6. Kết luận

Bài thực hành giúp em làm quen với quy trình xây dựng, huấn luyện và đánh giá các mô hình phân loại chuỗi. Kết quả hiện tại cho thấy mô hình cơ sở TF-IDF kết hợp Logistic Regression hoạt động tốt hơn các mô hình học sâu trong lần chạy này.

Bước tiếp theo là kiểm tra quy trình thực nghiệm và huấn luyện thêm để có thể so sánh công bằng hơn giữa các mô hình.
