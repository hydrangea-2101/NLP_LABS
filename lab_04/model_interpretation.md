# Phân tích trọng số mô hình Logistic Regression với TF-IDF

## 1. Mục tiêu

Mục tiêu của phần phân tích này là tìm hiểu cách mô hình Logistic Regression sử dụng các đặc trưng TF-IDF để phân loại đánh giá phim thành hai lớp: tích cực (positive) và tiêu cực (negative).

Thông qua việc kiểm tra trọng số của các đặc trưng, chúng ta có thể xác định những từ có mối liên hệ mạnh với từng lớp cảm xúc và đánh giá xem các từ này có hợp lý đối với bài toán phân loại cảm xúc hay không.

## 2. Phương pháp

Sau khi huấn luyện mô hình Logistic Regression trên các đặc trưng TF-IDF, trọng số của mô hình được lấy từ `model.coef_[0]`. Tên các đặc trưng tương ứng được lấy bằng `tfidf.get_feature_names_out()`.

Các đặc trưng được sắp xếp theo trọng số để tìm ra 20 đặc trưng có trọng số dương cao nhất và 20 đặc trưng có trọng số âm thấp nhất.

Thứ tự các lớp của mô hình là:

```python
['negative', 'positive']
```

Với mô hình Logistic Regression phân loại nhị phân này, trọng số dương cho thấy đặc trưng có xu hướng góp phần đưa dự đoán về lớp tích cực, trong khi trọng số âm cho thấy đặc trưng có xu hướng góp phần đưa dự đoán về lớp tiêu cực.

## 3. Các đặc trưng có trọng số dương cao nhất

Bảng dưới đây trình bày 20 đặc trưng có trọng số dương cao nhất của mô hình.

| Từ | Trọng số |
|---|---:|
| great | 8.511208 |
| excellent | 7.066828 |
| best | 5.507745 |
| perfect | 5.435190 |
| wonderful | 4.879188 |
| amazing | 4.863046 |
| loved | 4.770856 |
| brilliant | 4.380973 |
| enjoyed | 4.274907 |
| fun | 4.265831 |
| hilarious | 4.213025 |
| highly | 4.054009 |
| well | 4.023263 |
| superb | 3.990895 |
| love | 3.923892 |
| today | 3.892830 |
| favorite | 3.756780 |
| fantastic | 3.658569 |
| dvd | 3.635585 |
| definitely | 3.570205 |

## 4. Các đặc trưng có trọng số âm thấp nhất

Bảng dưới đây trình bày 20 đặc trưng có trọng số âm thấp nhất của mô hình.

| Từ | Trọng số |
|---|---:|
| worst | -10.462510 |
| bad | -9.102099 |
| awful | -7.934949 |
| waste | -7.666358 |
| boring | -6.782349 |
| poor | -6.332057 |
| terrible | -6.008926 |
| nothing | -5.756811 |
| horrible | -5.202991 |
| worse | -5.134600 |
| dull | -4.972088 |
| poorly | -4.850168 |
| stupid | -4.548137 |
| fails | -4.388295 |
| minutes | -4.368554 |
| annoying | -4.288387 |
| no | -4.267186 |
| instead | -4.228249 |
| disappointing | -4.170951 |
| unfortunately | -4.102950 |

## 5. Diễn giải kết quả và hạn chế

### 5.1. Các đặc trưng này có hợp lý với bài toán không?

Nhìn chung, các trọng số mà mô hình học được khá hợp lý đối với bài toán phân loại cảm xúc trong đánh giá phim.

Những từ như `great`, `excellent`, `wonderful` và `amazing` có trọng số dương cao. Đây là những từ thường được sử dụng để thể hiện sự khen ngợi hoặc đánh giá tích cực. Ngược lại, các từ như `worst`, `bad`, `awful` và `boring` có trọng số âm, phù hợp với cách chúng thường được sử dụng để thể hiện sự không hài lòng hoặc đánh giá tiêu cực về một bộ phim.

Điều này cho thấy mô hình đã học được một số đặc trưng từ vựng có ích cho nhiệm vụ phân loại cảm xúc.

Tuy nhiên, không nên chỉ dựa vào một từ có trọng số cao để kết luận cảm xúc của toàn bộ bài đánh giá. Dự đoán cuối cùng phụ thuộc vào sự kết hợp của các đặc trưng xuất hiện trong văn bản và giá trị TF-IDF tương ứng của chúng. Chẳng hạn, từ `great` thường mang ý nghĩa tích cực, nhưng trong câu "not great", từ phủ định `not` có thể làm thay đổi ý nghĩa của cả cụm từ.

Ngoài ra, một số từ như `today`, `dvd` và `minutes` không thể hiện cảm xúc rõ ràng khi đứng riêng lẻ. Việc chúng xuất hiện trong danh sách các đặc trưng có trọng số lớn có thể phản ánh những mối liên hệ mà mô hình học được từ dữ liệu huấn luyện. Những mối liên hệ này không có nghĩa là bản thân các từ đó luôn mang cảm xúc tích cực hoặc tiêu cực.

### 5.2. Hạn chế khi phân tích trọng số

Mặc dù việc kiểm tra trọng số giúp hiểu rõ hơn về các đặc trưng mà mô hình đã học, phương pháp này vẫn có một số hạn chế:

- **Khả năng nắm bắt ngữ cảnh còn hạn chế:** TF-IDF biểu diễn văn bản thông qua các đặc trưng số, nhưng không trực tiếp thể hiện đầy đủ ý nghĩa của câu hoặc thứ tự từ.
- **Khó xử lý câu phủ định và cảm xúc hỗn hợp:** Những câu như "not great" hoặc các bài đánh giá vừa khen vừa chê có thể khiến mô hình dự đoán sai.
- **Có thể học các mối liên hệ đặc thù của dữ liệu:** Một số từ có trọng số lớn vì chúng thường xuất hiện trong một loại đánh giá nhất định trong tập huấn luyện, chứ không nhất thiết vì chúng trực tiếp thể hiện cảm xúc.
- **Trọng số không phải xác suất:** Trọng số dương hoặc âm thể hiện hướng liên hệ của đặc trưng với lớp dự đoán. Không thể diễn giải trực tiếp một trọng số thành xác suất một bài đánh giá thuộc lớp tích cực hoặc tiêu cực.

Vì vậy, việc kiểm tra trọng số giúp cung cấp bằng chứng về những gì mô hình đã học, nhưng không thể giải thích đầy đủ mọi dự đoán hoặc chứng minh rằng mô hình thực sự hiểu ý nghĩa của bài đánh giá.

## 6. Kết luận

Kết quả phân tích cho thấy mô hình Logistic Regression kết hợp TF-IDF đã học được nhiều mối liên hệ hợp lý giữa từ ngữ và cảm xúc trong đánh giá phim. Các từ thể hiện sự khen ngợi thường có trọng số dương, trong khi các từ thể hiện sự chê bai thường có trọng số âm.

Tuy nhiên, một số đặc trưng khó diễn giải hơn, và trọng số của từng từ không thể phản ánh đầy đủ ngữ cảnh, câu phủ định hoặc cảm xúc trái chiều trong một bài đánh giá.

Do đó, việc phân tích trọng số nên được kết hợp với phân tích các mẫu dự đoán đúng và dự đoán sai để hiểu rõ hơn về khả năng cũng như hạn chế của mô hình.
