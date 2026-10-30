# Phân tích kết quả thực nghiệm

## 1. Mục tiêu đánh giá

Thí nghiệm nhằm so sánh hiệu quả phân loại cảm xúc của bốn mô hình:

- TF-IDF kết hợp Logistic Regression.
- RNN.
- LSTM.
- GRU.

Các chỉ số được sử dụng gồm Accuracy, Precision, Recall và F1-score. Ngoài ra, thời gian huấn luyện và Test Loss được dùng để quan sát thêm hiệu suất của mô hình.

## 2. Bảng kết quả

| Mô hình | Accuracy | Precision | Recall | F1-score | Thời gian huấn luyện (giây) | Test Loss |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0,8500 | 0,8448 | 0,8575 | 0,8511 | 0,560 | — |
| RNN | 0,5313 | 0,5289 | 0,5725 | 0,5498 | 1,524 | 0,6924 |
| LSTM | 0,5050 | 0,5060 | 0,4250 | 0,4620 | 3,592 | 0,6947 |
| GRU | 0,5050 | 0,5055 | 0,4625 | 0,4830 | 2,886 | 0,6959 |

## 3. Nhận xét kết quả

### 3.1. TF-IDF kết hợp Logistic Regression

Mô hình đạt Accuracy 85,00% và F1-score khoảng 85,11%, cao nhất trong bốn mô hình ở lần thực nghiệm này.

Kết quả cho thấy phương pháp biểu diễn văn bản bằng TF-IDF kết hợp với Logistic Regression là một mô hình cơ sở hiệu quả đối với bài toán phân loại cảm xúc IMDB.

### 3.2. Mô hình RNN

RNN đạt Accuracy khoảng 53,13% và F1-score khoảng 54,98%.

Hiệu suất hiện tại còn thấp so với mô hình cơ sở. Tuy nhiên, RNN mới được huấn luyện một epoch nên cần thêm thí nghiệm trước khi đưa ra kết luận cuối cùng.

### 3.3. Mô hình LSTM

LSTM đạt Accuracy 50,50% và F1-score khoảng 46,20%.

Test Loss khoảng 0,6947, gần mức 0,693. Đây là dấu hiệu cho thấy mô hình chưa thể hiện khả năng phân loại tốt trong lần chạy hiện tại.

### 3.4. Mô hình GRU

GRU đạt Accuracy 50,50% và F1-score khoảng 48,30%.

Kết quả cũng cho thấy mô hình chưa phân loại tốt trong lần thực nghiệm này. Test Loss khoảng 0,6959, do đó cần kiểm tra thêm quá trình huấn luyện và dự đoán.

## 4. So sánh tổng quát

Trong lần chạy hiện tại, TF-IDF kết hợp Logistic Regression cho kết quả tốt nhất và có thời gian huấn luyện ngắn nhất.

Ba mô hình học sâu có kết quả gần mức đoán ngẫu nhiên đối với bài toán hai lớp. Tuy nhiên, không thể từ đó kết luận rằng RNN, LSTM hay GRU vốn kém hiệu quả hơn TF-IDF. Các mô hình mới được huấn luyện một epoch và cần được kiểm tra thêm.

## 5. Hướng cải thiện

1. Kiểm tra cách ánh xạ nhãn tích cực và tiêu cực.
2. Kiểm tra dữ liệu đầu vào và đầu ra của mô hình.
3. Xác nhận hàm mất mát phù hợp với định dạng đầu ra.
4. Kiểm tra việc thêm padding, cắt ngắn chuỗi và sử dụng độ dài chuỗi.
5. Huấn luyện trong nhiều epoch hơn và theo dõi kết quả trên tập Validation.
6. Thử nghiệm các độ dài chuỗi khác nhau khi quy trình huấn luyện đã được xác nhận là chính xác.

## 6. Kết luận

Kết quả thực nghiệm hiện tại cho thấy TF-IDF kết hợp Logistic Regression là mô hình có hiệu suất tốt nhất trong số các mô hình được so sánh. Các mô hình RNN, LSTM và GRU chưa đạt kết quả mong muốn trong lần chạy thử nghiệm này.

Cần tiếp tục kiểm tra quy trình xử lý dữ liệu và huấn luyện trước khi đánh giá đầy đủ khả năng của các mô hình học sâu.
