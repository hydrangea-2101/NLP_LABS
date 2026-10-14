# Prediction Before Experiment — LAB 04

## Prediction 1: TF-IDF + Logistic Regression có khả năng hoạt động tốt không?

**Prediction:** Có khả năng hoạt động tốt.

**Reason:** TF-IDF chuyển văn bản thành các đặc trưng số dựa trên mức độ quan trọng của từ. Những đặc trưng này có thể giúp Logistic Regression học cách phân loại review theo hai nhãn positive và negative.

**Confidence:** Trung bình.

## Prediction 2: Average Word2Vec có chắc chắn tốt hơn TF-IDF không?

**Prediction:** Không chắc chắn tốt hơn.

**Reason:** Việc lấy trung bình các vector từ không đảm bảo giữ được thứ tự và cấu trúc của câu. Ví dụ, “Tôi thích bạn” và “Bạn thích tôi” sử dụng cùng các từ nhưng thể hiện những quan hệ khác nhau. Vì vậy, Average Word2Vec có thể làm mất một phần thông tin quan trọng của văn bản.

**Confidence:** Cao.

## Prediction 3: Nếu tăng vocabulary size thì Accuracy chắc chắn tăng không?

**Prediction:** Không chắc chắn Accuracy sẽ tăng.

**Reason:** Việc đưa thêm nhiều từ vào vocabulary không đảm bảo cung cấp thêm thông tin hữu ích. Một số từ như `a`, `an`, `the` thường ít có khả năng phân biệt cảm xúc, nên việc giữ thêm các từ ít hữu ích có thể không cải thiện kết quả và có thể làm tăng nhiễu.

**Confidence:** Trung bình.

## Prediction 4: Nếu bỏ stopwords thì performance chắc chắn tăng không?

**Prediction:** Không chắc chắn performance sẽ tăng.

**Reason:** Không phải cứ loại bỏ stopwords thì hiệu suất mô hình sẽ tăng. Một số từ trong danh sách stopwords vẫn có thể mang ý nghĩa quan trọng đối với việc xác định cảm xúc. Nếu loại bỏ những từ này, ý nghĩa của câu có thể thay đổi hoặc thông tin cần thiết cho mô hình có thể bị mất.

**Confidence:** Trung bình.

## Prediction 5: Accuracy và F1 có thể cho kết luận khác nhau không?

**Prediction:** Có, chúng có thể cho kết luận khác nhau.

**Reason:** Một mô hình dự đoán tốt lớp chiếm đa số nhưng bỏ sót phần lớn mẫu thuộc lớp ít hơn vẫn có thể đạt Accuracy cao. F1 kết hợp Precision và Recall nên có thể phản ánh những hạn chế đó rõ hơn, tùy vào cách tính F1.

**Confidence:** Trung bình.