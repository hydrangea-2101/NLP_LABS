# Phân tích lỗi dự đoán

## 1. Mục tiêu

Phân tích lỗi nhằm tìm hiểu những trường hợp mô hình dự đoán sai và xác định các vấn đề có thể ảnh hưởng đến kết quả phân loại cảm xúc.

Trong lần chạy hiện tại, RNN, LSTM và GRU có Accuracy xấp xỉ 50–53%, thấp hơn đáng kể so với TF-IDF kết hợp Logistic Regression. Vì vậy, cần kiểm tra cả các dự đoán cụ thể và quy trình huấn luyện.

## 2. Một số trường hợp dự đoán

Các ví dụ dưới đây được lấy từ kết quả chạy notebook.

| Văn bản | Nhãn dự đoán | Xác suất lớp tích cực |
|---|---|---:|
| The acting was excellent. | Tích cực | 0,505 |
| The plot was boring and the acting was terrible. | Tích cực | 0,510 |
| I thought it would be bad, but it was surprisingly good. | Tiêu cực | 0,497 |

Những ví dụ trên cho thấy mô hình có thể đưa ra dự đoán chưa phù hợp với sắc thái cảm xúc của câu.

Đặc biệt, câu “The plot was boring and the acting was terrible.” chứa nhiều từ mang ý nghĩa tiêu cực nhưng lại được dự đoán là tích cực. Đây là một trường hợp cần kiểm tra thêm.

Các ví dụ trên chỉ phản ánh một số trường hợp quan sát được, chưa đủ để đại diện cho toàn bộ lỗi của mô hình.

## 3. Những vấn đề cần kiểm tra

### 3.1. Cách mã hóa nhãn

Cần xác nhận nhãn tích cực và tiêu cực được ánh xạ thống nhất trong toàn bộ quy trình.

Nếu nhãn được mã hóa không nhất quán giữa dữ liệu, quá trình huấn luyện và bước tính toán kết quả, mô hình có thể học hoặc đánh giá sai.

### 3.2. Quy trình xử lý văn bản

Cần kiểm tra các bước:

- Chuyển văn bản thành chuỗi token.
- Ánh xạ token sang số nguyên.
- Xử lý token chưa xuất hiện trong từ điển.
- Thêm padding cho chuỗi ngắn.
- Cắt ngắn chuỗi dài.
- Đảm bảo dữ liệu đầu vào có đúng định dạng mô hình yêu cầu.

Đây là những điểm cần kiểm tra, chưa thể khẳng định chúng là nguyên nhân gây ra kết quả thấp.

### 3.3. Quá trình huấn luyện

Các mô hình RNN, LSTM và GRU mới được huấn luyện một epoch trong lần chạy hiện tại.

Do đó, cần quan sát thêm Loss và các chỉ số trên tập huấn luyện, tập kiểm định qua nhiều epoch để đánh giá mô hình có học được đặc trưng cảm xúc hay không.

### 3.4. Đầu ra và hàm mất mát

Cần xác nhận đầu ra của mô hình phù hợp với hàm mất mát được sử dụng. Đồng thời, cần kiểm tra kích thước tensor, cách tính xác suất và quy tắc chuyển đầu ra thành nhãn dự đoán.

### 3.5. Khả năng xử lý ngữ cảnh

Một số câu chứa cả từ tích cực lẫn tiêu cực, hoặc có cấu trúc chuyển ý như “but”. Những trường hợp này yêu cầu mô hình hiểu ngữ cảnh thay vì chỉ dựa vào từng từ riêng lẻ.

Tuy nhiên, chưa đủ dữ liệu để kết luận rằng đây là nguyên nhân chính khiến các mô hình hiện tại hoạt động kém.

## 4. Kế hoạch kiểm tra và cải thiện

1. Kiểm tra nhãn của một số mẫu dữ liệu sau khi tiền xử lý.
2. Kiểm tra một vài mẫu sau khi mã hóa token và padding.
3. Đối chiếu đầu ra mô hình với nhãn thực tế.
4. Theo dõi Loss và các chỉ số trên tập huấn luyện, tập kiểm định.
5. Huấn luyện thêm epoch và so sánh kết quả.
6. Tạo danh sách các mẫu dự đoán sai để phân tích có hệ thống.

## 5. Kết luận

Kết quả hiện tại cho thấy các mô hình học sâu chưa phân loại cảm xúc hiệu quả trong lần chạy thử nghiệm. Phân tích lỗi đã xác định được một số dự đoán chưa phù hợp và các bước cần kiểm tra tiếp theo.

Chưa thể khẳng định nguyên nhân cụ thể khi chưa thực hiện các bước kiểm chứng. Việc phân tích cần dựa trên dữ liệu, đầu ra mô hình và kết quả của những lần chạy tiếp theo.
