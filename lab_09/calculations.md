# LAB 09: BÀI TẬP TÍNH TOÁN & CƠ SỞ LÝ THUYẾT (CALCULATIONS)

---

## 1. Exercise 1: Sơ đồ gán nhãn BIO (BIO Tagging Scheme)

### 1.1. Đề bài
Cho câu văn bản:  
> *"Elon Musk founded SpaceX in California."*

Hãy xác định các thực thể có tên (Named Entities) theo tập nhãn chuẩn CoNLL-2003 và thực hiện gán nhãn BIO cấp từ (Word-level BIO tagging).

### 1.2. Phân tích thực thể và Bảng gán nhãn BIO
Các thực thể có tên trong câu gồm:
1. **"Elon Musk"**: Thực thể chỉ người (Person - `PER`), kéo dài qua 2 token.
2. **"SpaceX"**: Thực thể chỉ tổ chức / doanh nghiệp (Organization - `ORG`), gồm 1 token.
3. **"California"**: Thực thể chỉ địa danh / vị trí địa lý (Location - `LOC`), gồm 1 token.
4. Các từ còn lại (*"founded"*, *"in"*, dấu chấm *"."*) không phải thực thể, thuộc nhãn ngoài (`O`).

| Vị trí $t$ | Token (Word) | Nhãn BIO | Giải thích chi tiết |
|:---:|:---|:---:|:---|
| 1 | `Elon` | **`B-PER`** | Bắt đầu (Begin) của thực thể Người (`PER`). |
| 2 | `Musk` | **`I-PER`** | Bên trong (Inside) tiếp nối thực thể Người (`PER`) bắt đầu từ "Elon". |
| 3 | `founded` | **`O`** | Động từ thường, không thuộc thực thể nào (Outside). |
| 4 | `SpaceX` | **`B-ORG`** | Bắt đầu của thực thể Tổ chức (`ORG`) đơn token. |
| 5 | `in` | **`O`** | Giới từ thường, không thuộc thực thể nào (Outside). |
| 6 | `California` | **`B-LOC`** | Bắt đầu của thực thể Vị trí địa lý (`LOC`) đơn token. |
| 7 | `.` | **`O`** | Dấu chấm kết câu, không thuộc thực thể (Outside). |

---

## 2. Exercise 2: Phân định Ranh giới Thực thể (Entity Boundary Identification)

### 2.1. Đề bài
Cho cụm từ chỉ tổ chức giáo dục:  
> *"University of California Berkeley"* (được xác định là một thực thể Organization duy nhất).

1. Hãy gán nhãn BIO chi tiết cho từng từ.
2. Giải thích tại sao việc xác định ranh giới thực thể (Entity Boundary) bắt buộc phải phân biệt giữa tiền tố `B-` (Begin) và `I-` (Inside), thay vì chỉ gán nhãn loại thực thể đơn thuần (như `ORG`, `LOC`, `PER`).

### 2.2. Bảng gán nhãn BIO cho cụm thực thể

| Token (Word) | Nhãn BIO | Vai trò trong Span thực thể |
|:---|:---:|:---|
| `University` | **`B-ORG`** | Điểm bắt đầu (Boundary Start) của thực thể Organization. |
| `of` | **`I-ORG`** | Thành phần giới từ bên trong span của Organization. |
| `California` | **`I-ORG`** | Thành phần tên bang bên trong span của Organization (không phải B-LOC vì nằm trong tên trường). |
| `Berkeley` | **`I-ORG`** | Thành phần phân hiệu bên trong span của Organization. |

### 2.3. Tại sao bắt buộc phải có sự phân biệt giữa `B-` và `I-`?

Nếu hệ thống chỉ sử dụng các nhãn loại thực thể thô (ví dụ: `PER`, `ORG`, `LOC`, `MISC`, `O`) mà không có tiền tố phân định ranh giới `B-` và `I-`, hệ thống sẽ gặp phải **vấn đề nhập nhằng ranh giới không thể giải quyết (Boundary Ambiguity)** khi xuất hiện **hai thực thể cùng loại đứng liền kề nhau**.

#### Minh chứng phản ví dụ kinh điển:
Xét 2 câu văn sau:
* **Câu A (1 thực thể nhiều từ):**  
  > *"She flew to [New York] yesterday."* $\to$ 1 thực thể `LOC` kéo dài 2 từ.
* **Câu B (2 thực thể riêng biệt đứng cạnh nhau không có dấu phẩy):**  
  > *"She flew from [Paris] [London] tickets were cheap."* (hoặc trong tiêu đề báo: *"Jane visited [Paris] [London] on Monday"*).

1. **Nếu không có tiền tố BIO (chỉ dùng nhãn thô `LOC`):**
   * Câu A: `New` (`LOC`), `York` (`LOC`) $\implies$ Chuỗi nhãn: `[LOC, LOC]`.
   * Câu B: `Paris` (`LOC`), `London` (`LOC`) $\implies$ Chuỗi nhãn: `[LOC, LOC]`.
   * **Hậu quả:** Bộ giải mã (Decoder / Post-processing) hoàn toàn bất lực trong việc xác định liệu `[LOC, LOC]` là **một địa danh duy nhất gồm 2 từ** ("Paris London") hay là **hai địa danh riêng biệt** ("Paris" và "London")!

2. **Khi sử dụng sơ đồ BIO (IOB2 Scheme):**
   * Câu A: `New` (**`B-LOC`**), `York` (**`I-LOC`**) $\implies$ Trình phân tích biết chắc chắn đây là **một thực thể duy nhất** bắt đầu tại "New" và kéo dài qua "York".
   * Câu B: `Paris` (**`B-LOC`**), `London` (**`B-LOC`**) $\implies$ Token "London" mang nhãn **`B-LOC`** phát tín hiệu dứt khoát rằng thực thể trước đã đóng lại và một thực thể `LOC` hoàn toàn mới đang bắt đầu.

> **Kết luận:** Tiền tố `B-` và `I-` biến bài toán phân đoạn văn bản (Text Segmentation) và phân loại ngữ nghĩa (Semantic Tagging) thành một bài toán gán nhãn chuỗi duy nhất, cho phép mô hình xác định chính xác điểm bắt đầu, điểm tiếp nối và ranh giới kết thúc của mọi thực thể đa token (multi-token entities).

---

## 3. Exercise 3: Tính toán Kích thước Tensor (Tensor Shapes) & Số lượng Dự đoán

### 3.1. Thiết lập tham số
Cho một mô hình Transformer Token Classification với các siêu tham số chuẩn:
* Batch size: $B = 8$
* Maximum Sequence Length: $T = 128$
* Hidden Dimension: $d = 768$ (kích thước vector biểu diễn ẩn của BERT-Base)
* Số lượng lớp phân loại (Classes): $C = 9$ (gồm `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`)

### 3.2. Tính toán kích thước các Tensor qua từng tầng

#### a. Hidden States Tensor ($H$)
Đầu ra từ tầng Transformer Encoder cuối cùng chứa vector biểu diễn ngữ cảnh cho mọi vị trí token trong toàn bộ batch:
$$\text{Shape}(H) = [B, T, d] = [8, 128, 768]$$
* **Số phần tử:** $8 \times 128 \times 768 = 786,432$ giá trị thực.

#### b. Token Classification Head (Linear Layer)
Tầng phân loại tuyến tính thực hiện phép chiếu từ không gian ẩn sang không gian lớp dự đoán cho từng vị trí token:
$$\text{Logits} = H W^T + b$$
Trong đó:
* Ma trận trọng số: $W \in \mathbb{R}^{C \times d} \implies \text{Shape}(W) = [9, 768]$ (gồm $9 \times 768 = 6,912$ tham số).
* Vector độ lệch: $b \in \mathbb{R}^C \implies \text{Shape}(b) = [9]$ (gồm 9 tham số).
* Kích thước Classification Logits Tensor ($Z$):
$$\text{Shape}(Z) = [B, T, C] = [8, 128, 9]$$
* **Số phần tử:** $8 \times 128 \times 9 = 9,216$ logits.

#### c. Ground-truth Labels Tensor ($Y$)
Tensor nhãn mục tiêu chứa ID lớp tương ứng (hoặc giá trị `-100` cho các vị trí bị bỏ qua):
$$\text{Shape}(Y) = [B, T] = [8, 128]$$
* **Số phần tử:** $8 \times 128 = 1,024$ nhãn số nguyên.

#### d. Tổng số dự đoán cấp token (Total Token Predictions)
Trên một batch huấn luyện đơn lẻ, mô hình thực hiện:
$$N_{\text{predictions}} = B \times T = 8 \times 128 = 1,024 \text{ dự đoán riêng biệt.}$$

---

## 4. Hàm mất mát Token Classification & Cơ chế Masking (`ignore_index = -100`)

### 4.1. Công thức Cross-Entropy Loss cấp Token
Với mỗi token $t \in \{1, \dots, T\}$ của mẫu thứ $b \in \{1, \dots, B\}$, phân phối xác suất dự đoán qua Softmax là:
$$P(y_{b,t} = c \mid \mathbf{x}_{b}) = \frac{\exp(z_{b,t,c})}{\sum_{k=1}^C \exp(z_{b,t,k})}$$

Do các vị trí padding (`[PAD]`), token đặc biệt (`[CLS]`, `[SEP]`), và các subwords tiếp theo (theo Strategy A) không được tính vào hàm mất mát, PyTorch sử dụng nhãn đặc biệt $\text{IGNORE\_INDEX} = -100$.

Hàm mất mát Cross-Entropy trung bình trên toàn batch được định nghĩa:
$$\mathcal{L} = -\frac{1}{N_{\text{active}}} \sum_{b=1}^B \sum_{t=1}^T \mathbb{I}(y_{b,t} \neq -100) \cdot \log P(y_{b,t} \mid \mathbf{x}_b)$$

Trong đó:
* $\mathbb{I}(\cdot)$ là hàm chỉ thị (Indicator function), nhận giá trị 1 nếu điều kiện đúng và 0 nếu sai.
* $N_{\text{active}} = \sum_{b=1}^B \sum_{t=1}^T \mathbb{I}(y_{b,t} \neq -100)$ là tổng số token hợp lệ thực tế tham gia cập nhật gradient.

### 4.2. Chứng minh Gradient bằng 0 cho các vị trí bị Masked
Xét một vị trí token $(b, t)$ có nhãn $y_{b,t} = -100$.
Theo định nghĩa của hàm mất mát có điều kiện chỉ thị:
$$\frac{\partial \mathcal{L}}{\partial z_{b,t,c}} = 0 \quad \forall c \in \{1, \dots, C\}$$

Do đạo hàm riêng của Loss theo logit tại vị trí bị mask triệt tiêu hoàn toàn về 0, theo quy tắc dây chuyền (Chain Rule):
$$\frac{\partial \mathcal{L}}{\partial \theta_{\text{encoder}}} = \sum_{b, t} \frac{\partial \mathcal{L}}{\partial z_{b,t}} \frac{\partial z_{b,t}}{\partial h_{b,t}} \frac{\partial h_{b,t}}{\partial \theta_{\text{encoder}}} = \sum_{(b,t): y_{b,t} \neq -100} \frac{\partial \mathcal{L}}{\partial z_{b,t}} \frac{\partial z_{b,t}}{\partial h_{b,t}} \frac{\partial h_{b,t}}{\partial \theta_{\text{encoder}}}$$

**Hệ quả trực tiếp:** Các token padding và các subwords bổ sung không bao giờ làm nhiễu tín hiệu học của mô hình, đảm bảo gradient chỉ tập trung tối ưu hóa các điểm biểu diễn thực sự có ý nghĩa ngôn ngữ học.
