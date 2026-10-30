# LAB 04 — Text Classification

## 1. Giới thiệu

Bài lab thực hiện bài toán phân loại cảm xúc (*Sentiment Classification*) trên các bài đánh giá phim bằng tiếng Anh. Mục tiêu là xây dựng mô hình phân loại đánh giá thành hai lớp: `positive` và `negative`.

Thí nghiệm so sánh hai phương pháp biểu diễn văn bản:

- **TF-IDF**
- **Average Word2Vec**

Cả hai phương pháp đều sử dụng Logistic Regression làm bộ phân loại.

## 2. Dataset

Sử dụng bộ dữ liệu IMDB Movie Reviews, gồm 50.000 bài đánh giá phim với hai cột chính:

- `review`: nội dung bài đánh giá.
- `sentiment`: nhãn cảm xúc (`positive` hoặc `negative`).

Dữ liệu được chia thành ba tập theo tỷ lệ 70% / 15% / 15%, sử dụng stratified split để duy trì tỷ lệ nhãn giữa các tập:

| Tập dữ liệu | Số mẫu |
|---|---:|
| Train | 35.000 |
| Validation | 7.500 |
| Test | 7.500 |

## 3. Phương pháp

### TF-IDF + Logistic Regression

TF-IDF biểu diễn văn bản bằng các đặc trưng số dựa trên mức độ quan trọng của từ. Logistic Regression được huấn luyện trên các vector TF-IDF để dự đoán cảm xúc.

### Average Word2Vec + Logistic Regression

Word2Vec được huấn luyện trên văn bản của tập Train. Vector của một bài đánh giá được tạo bằng cách lấy trung bình vector của các từ có trong từ vựng Word2Vec. Logistic Regression sử dụng vector trung bình này để phân loại cảm xúc.

## 4. Kết quả

Kết quả trên tập Validation:

| Representation | Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| TF-IDF | Logistic Regression | 0.8903 | 0.8903 | 0.8903 | 0.8903 |
| Average Word2Vec | Logistic Regression | 0.8416 | 0.8416 | 0.8416 | 0.8416 |

Mô hình TF-IDF + Logistic Regression được lựa chọn để đánh giá trên tập Test. Accuracy trên tập Test đạt **0.9015**.

Các kết quả chi tiết được lưu trong `results.csv`.

## 5. Cấu trúc thư mục

- `classification.ipynb`: notebook chứa quá trình khám phá dữ liệu, tiền xử lý, huấn luyện và đánh giá mô hình.
- `preprocessing.py`: hàm tokenization dùng cho Word2Vec.
- `results.csv`: bảng so sánh kết quả thực nghiệm.
- `confusion_matrix.png`: hình ảnh ma trận nhầm lẫn.
- `error_analysis.md`: phân tích các ví dụ dự đoán đúng và sai.
- `model_interpretation.md`: phân tích trọng số của mô hình Logistic Regression sử dụng TF-IDF.
- `calculations.pdf`: phần tính toán lý thuyết.
- `prediction.md`: phần dự đoán theo yêu cầu bài lab.
- `reflection.md`: phần tự đánh giá và rút kinh nghiệm.

## 6. Cách chạy

1. Cài đặt các thư viện cần thiết trong môi trường Python.
2. Đảm bảo file dataset nằm đúng đường dẫn mà notebook sử dụng.
3. Mở và chạy các cell trong `classification.ipynb` theo thứ tự.
4. Kiểm tra các file kết quả được tạo trong thư mục bài lab.

Các thư viện chính gồm pandas, NumPy, scikit-learn và gensim.

## 7. Công cụ AI hỗ trợ

AI được sử dụng để hỗ trợ giải thích khái niệm, gỡ lỗi, hướng dẫn sử dụng thư viện và kiểm tra cách tổ chức project. Các bước thực nghiệm và kết quả được chạy trong môi trường Python của người học. Người học cần tự kiểm tra, hiểu và chịu trách nhiệm với nội dung nộp.