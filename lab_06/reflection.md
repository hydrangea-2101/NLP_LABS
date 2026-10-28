# LAB 06: BÁO CÁO PHẢN TƯ / ĐÚC KẾT HỌC TẬP (REFLECTION)
### Về Query (Q), Key (K) và Value (V)?
- **Query ($Q$):** Biểu diễn thông tin mà vị trí hiện tại đang "tìm kiếm" hoặc "đặt câu hỏi" đối với toàn bộ chuỗi văn bản.
- **Key ($K$):** Đóng vai trò như các "nhãn mục tiêu" hoặc đặc trưng mô tả của từng token trong câu, dùng để so khớp độ tương đồng với Query.
- **Value ($V$):** Là thông tin nội dung thực sự sẽ được trích xuất và kết hợp để tạo ra vector ngữ cảnh mới.
- **Phép loại suy:** Tương tự như cơ chế tìm kiếm trong cơ sở dữ liệu hoặc công cụ tìm kiếm: bạn đưa vào một câu truy vấn ($Q$), hệ thống so khớp $Q$ với các tiêu đề mục từ khóa ($K$), sau đó trả về nội dung của các trang web tương ứng ($V$) theo mức độ liên quan.
- **Trong Self-Attention:** Cả $Q, K, V$ đều được sinh ra từ cùng một chuỗi biểu diễn đầu vào $X$ thông qua ba phép chiếu tuyến tính độc lập: $Q = X W_Q$, $K = X W_K$, $V = X W_V$.

### 2. Vì sao cần hàm Softmax trong Attention?
- Hàm Softmax chuyển đổi các điểm số thô vô hướng ($s_i = q^T k_i$, có thể âm, dương hoặc lớn tùy ý) thành một phân phối xác suất hợp lệ:
  $$\alpha_i \ge 0 \quad \text{và} \quad \sum_i \alpha_i = 1$$
- Nhờ Softmax, các trọng số $\alpha_i$ đóng vai trò là các hệ số kết hợp lồi (convex combination weights) trong phép tính vector ngữ cảnh $c = \sum_i \alpha_i v_i$. Điều này đảm bảo độ lớn của vector đầu ra không bị bùng nổ theo chiều dài chuỗi và có thể tính gradient liên tục để lan truyền ngược (backpropagation).

### 3. Scaling bằng $\frac{1}{\sqrt{d_k}}$ giải quyết vấn đề gì?
- Khi số chiều của Key vector ($d_k$) tăng lên (ví dụ $d_k = 64$ hoặc $512$), nếu các thành phần của $q$ và $k$ là các biến ngẫu nhiên độc lập có kỳ vọng bằng 0 và phương sai bằng 1, thì tích vô hướng $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ sẽ có kỳ vọng bằng 0 và phương sai bằng $d_k$.
- Khi $d_k$ lớn, giá trị tích vô hướng có độ phân tán rất rộng, đẩy các điểm số score vào các vùng cực trị nơi hàm Softmax có độ dốc (đạo hàm) cực nhỏ (vùng bão hòa - saturation).
- Hậu quả: Hiện tượng triệt tiêu gradient (vanishing gradient) khiến mạng nơ-ron học rất chậm hoặc ngừng học.
- Chia cho $\sqrt{d_k}$ giúp kéo phương sai của các scores về lại mức 1, giữ cho hàm Softmax hoạt động trong vùng nhạy cảm của gradient.

### 4. Những gì quan sát được từ Attention Matrix?
- **Độ phức tạp và cấu trúc:** Ma trận Attention kích thước $T \times T$ phản ánh tương tác 2 chiều giữa mọi cặp token trong câu.
- **Phân phối:** Các token có thể phân phối sự chú ý đồng đều hoặc tập trung mạnh vào một vài từ khóa chủ chốt.
- **Hiện tượng Attention Sink:** Một số token hư từ (như dấu chấm, `<PAD>`, mạo từ *"the"*) đôi khi nhận trọng số khá lớn chỉ vì mô hình cần một "bồn chứa" để phân tán bớt trọng số khi không tìm thấy thông tin phù hợp ở các token khác.

### 5. Attention có chứng minh mô hình hiểu ngữ nghĩa hay reasoning không?
- **HOÀN TOÀN KHÔNG.**
- Cần phân biệt rõ:
  - *Attention Pattern:* Là giá trị số học được tính từ phép nhân ma trận tối ưu theo hàm mất mát trong không gian vector.
  - *Linguistic Reasoning:* Là khả năng suy luận logic và ngữ pháp theo nhận thức con người.
- Attention weights cao tại một từ chỉ cho thấy vector biểu diễn của từ đó có tương quan toán học với truy vấn, chứ không chứng minh mô hình "hiểu" tại sao từ đó lại quan trọng.

### 6. Độ phức tạp $O(T^2)$ và giới hạn của Self-Attention
- Self-Attention đòi hỏi tính toán tương tác giữa mọi cặp token, dẫn đến ma trận kích thước $T \times T$.
- Khi sequence length $T$ tăng từ $1.000 \to 10.000$, số phần tử tăng từ $10^6 \to 10^8$ (gấp 100 lần), tiêu tốn hàng trăm MB đến hàng GB bộ nhớ chỉ để lưu ma trận chú ý thô cho một lớp.
- Đây chính là nút thắt cổ chai về bộ nhớ và tính toán lớn nhất của kiến trúc Transformer khi mở rộng sang văn bản dài (Long context), dẫn đến các nghiên cứu về Linear Attention, FlashAttention hay Local Window Attention.

---
