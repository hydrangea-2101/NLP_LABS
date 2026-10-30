# Reflection — LAB 04: Text Classification

## 1. So sánh TF-IDF và Average Word2Vec

Qua bài lab, em nhận thấy TF-IDF và Average Word2Vec có cách biểu diễn văn bản khác nhau. TF-IDF biểu diễn review dựa trên các từ trong từ điển và trọng số của chúng, trong khi Average Word2Vec biểu diễn mỗi từ bằng một vector rồi lấy trung bình các vector để tạo thành vector đại diện cho cả review.

Trong thực nghiệm, TF-IDF kết hợp với Logistic Regression đạt Accuracy 0.8903 trên tập validation, cao hơn Average Word2Vec kết hợp với Logistic Regression, đạt 0.8416.

## 2. Hạn chế của Average Word2Vec

Theo em, một hạn chế của Average Word2Vec là việc lấy trung bình các word vector có thể làm mất thông tin về thứ tự từ trong câu. Vì vậy, mô hình có thể gặp khó khăn khi phân biệt những review có nhiều từ giống nhau nhưng mang ý nghĩa hoặc cảm xúc trái ngược nhau.

Ví dụ, hai câu “I like this movie” và “I don't like this movie” có nhiều từ chung nhưng mang cảm xúc khác nhau. Việc lấy trung bình các vector có thể làm giảm khả năng thể hiện rõ sự khác biệt giữa hai câu. Đây là một nguyên nhân có thể góp phần giải thích kết quả thấp hơn của Average Word2Vec trong bài lab, mặc dù chưa thể khẳng định đó là nguyên nhân duy nhất.

## 3. Hạn chế của TF-IDF và bài học rút ra

Em nhận thấy TF-IDF cũng có những hạn chế. Một review có thể chứa cả những từ mang sắc thái tích cực và tiêu cực, nhưng các từ riêng lẻ không phải lúc nào cũng thể hiện chính xác cảm xúc của toàn bộ văn bản.

Do đó, em cho rằng không thể khẳng định TF-IDF luôn tốt hơn Word2Vec chỉ dựa trên kết quả của bài lab này. Hiệu quả của mỗi phương pháp còn phụ thuộc vào dữ liệu, cách biểu diễn văn bản và mô hình phân loại được sử dụng.

Qua bài lab, em hiểu rõ hơn rằng việc lựa chọn phương pháp biểu diễn văn bản có ảnh hưởng đến kết quả phân loại cảm xúc. Đồng thời, khi phân tích kết quả thực nghiệm, cần xem xét cả ưu điểm, hạn chế của phương pháp và đặc điểm của dữ liệu, thay vì chỉ dựa vào một chỉ số đánh giá.