# LAB 05 – Mô hình chuỗi: RNN, LSTM và GRU

## 1. Giới thiệu

Bài thực hành này tìm hiểu và áp dụng các mô hình học sâu xử lý dữ liệu dạng chuỗi để giải quyết bài toán phân loại cảm xúc trong văn bản.

Dữ liệu sử dụng là bộ dữ liệu IMDB Movie Reviews, bao gồm các bài đánh giá phim được gán nhãn tích cực hoặc tiêu cực.

## 2. Mục tiêu

- Tìm hiểu cách biểu diễn văn bản dưới dạng chuỗi số nguyên.
- Hiểu cách sử dụng từ điển từ vựng, mã hóa từ và xử lý độ dài chuỗi.
- Xây dựng mô hình RNN, LSTM và GRU để phân loại cảm xúc.
- So sánh các mô hình học sâu với mô hình TF-IDF kết hợp Logistic Regression.
- Đánh giá mô hình bằng Accuracy, Precision, Recall và F1-score.
- Phân tích kết quả và tìm hiểu những hạn chế của các mô hình.

## 3. Dữ liệu

Bộ dữ liệu IMDB được chia thành ba tập:

| Tập dữ liệu | Số lượng mẫu |
|---|---:|
| Huấn luyện (Train) | 40.000 |
| Kiểm định (Validation) | 5.000 |
| Kiểm tra (Test) | 5.000 |
| **Tổng cộng** | **50.000** |

Tập huấn luyện có số lượng mẫu tích cực và tiêu cực cân bằng, mỗi lớp gồm 20.000 mẫu.

## 4. Các mô hình

### 4.1. TF-IDF kết hợp Logistic Regression

Đây là mô hình cơ sở dùng để phân loại văn bản. TF-IDF chuyển văn bản thành vector đặc trưng, sau đó Logistic Regression thực hiện dự đoán nhãn cảm xúc.

### 4.2. RNN

RNN xử lý chuỗi theo thứ tự từng bước và duy trì trạng thái ẩn để biểu diễn thông tin đã tiếp nhận.

### 4.3. LSTM

LSTM mở rộng RNN bằng cơ chế cổng và trạng thái ô nhớ, giúp kiểm soát việc lưu giữ và cập nhật thông tin trong chuỗi.

### 4.4. GRU

GRU sử dụng các cổng để điều chỉnh thông tin được giữ lại hoặc cập nhật, với cấu trúc đơn giản hơn LSTM.

## 5. Kết quả thực nghiệm

| Mô hình | Accuracy | F1-score |
|---|---:|---:|
| TF-IDF + Logistic Regression | 85,00% | 85,11% |
| RNN | 53,13% | 54,98% |
| LSTM | 50,50% | 46,20% |
| GRU | 50,50% | 48,30% |

Các mô hình RNN, LSTM và GRU trong lần chạy hiện tại mới được huấn luyện một epoch. Vì vậy, kết quả này chỉ phản ánh lần chạy thử nghiệm hiện tại, chưa đủ để kết luận về hiệu quả tối đa của từng kiến trúc.

## 6. Cấu trúc dự án

- `sequence_classification(1).ipynb`: Notebook thực hiện các bước xử lý dữ liệu, huấn luyện và đánh giá mô hình.
- `rnn_cell.py`: Kiểm chứng phép tính của một ô Simple RNN.
- `results.csv`: Lưu bảng kết quả đánh giá.
- `learning_curves.png`: Biểu đồ quá trình học của mô hình, nếu đã được tạo.
- `confusion_matrix.png`: Ma trận nhầm lẫn, nếu đã được tạo.
- `reports/results.md`: Phân tích kết quả thực nghiệm.
- `reports/error_analysis.md`: Phân tích lỗi dự đoán.
- `reports/reflection.md`: Tổng kết kiến thức và kinh nghiệm sau bài thực hành.

Tên notebook trong danh sách trên cần được điều chỉnh nếu file thực tế của em có tên khác.

## 7. Cài đặt môi trường

Cài đặt các thư viện được liệt kê trong `requirements.txt`:

```bash
pip install -r requirements.txt
```

Mở notebook bằng Jupyter Notebook hoặc JupyterLab và chạy các ô theo thứ tự.

## 8. Hướng phát triển

- Kiểm tra lại quy trình xử lý văn bản và mã hóa nhãn.
- Huấn luyện các mô hình RNN, LSTM và GRU trong nhiều epoch hơn.
- Theo dõi đồng thời kết quả trên tập huấn luyện và tập kiểm định.
- Thử nghiệm các độ dài chuỗi đầu vào khác nhau.
- Phân tích các trường hợp mô hình dự đoán sai để tìm hướng cải thiện.
