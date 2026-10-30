import numpy as np
import torch
import torch.nn as nn

class SimpleRNNCell:
    """
    Tự triển khai Simple RNN Cell từ đầu bằng NumPy.
    Công thức: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Khởi tạo trọng số ngẫu nhiên theo phân phối chuẩn
        np.random.seed(42)
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        x_t: shape (input_size, batch_size) hoặc (input_size, 1)
        h_prev: shape (hidden_size, batch_size) hoặc (hidden_size, 1)
        return: h_t shape (hidden_size, batch_size)
        """
        a_t = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
        h_t = np.tanh(a_t)
        return h_t

def test_rnn_cell_implementation():
    print("=== TESTING CUSTOM SIMPLE RNN CELL vs PYTORCH RNN CELL ===")
    input_size = 4
    hidden_size = 3
    batch_size = 1

    # Khởi tạo Custom Cell
    custom_cell = SimpleRNNCell(input_size=input_size, hidden_size=hidden_size)

    # Khởi tạo PyTorch RNNCell
    torch_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size, bias=True)

    # Đồng bộ trọng số từ PyTorch sang Custom Cell để so sánh chính xác
    with torch.no_grad():
        torch_cell.weight_ih.copy_(torch.tensor(custom_cell.W_xh, dtype=torch.float32))
        torch_cell.weight_hh.copy_(torch.tensor(custom_cell.W_hh, dtype=torch.float32))
        torch_cell.bias_ih.copy_(torch.tensor(custom_cell.b_h.squeeze(), dtype=torch.float32))
        torch_cell.bias_hh.fill_(0.0)  # PyTorch có 2 bias (ih và hh), ta đặt hh bias = 0

    # Input ngẫu nhiên
    x_t_np = np.random.randn(input_size, batch_size).astype(np.float32)
    h_prev_np = np.random.randn(hidden_size, batch_size).astype(np.float32)

    # Compute Custom Forward
    h_t_custom = custom_cell.forward(x_t_np, h_prev_np)

    # Compute PyTorch Forward
    x_t_torch = torch.tensor(x_t_np.T, dtype=torch.float32)
    h_prev_torch = torch.tensor(h_prev_np.T, dtype=torch.float32)
    h_t_torch = torch_cell(x_t_torch, h_prev_torch).detach().numpy().T

    # Đánh giá độ lệch (Absolute Error)
    diff = np.max(np.abs(h_t_custom - h_t_torch))
    print(f"Custom Cell Output:\n{h_t_custom.squeeze()}")
    print(f"PyTorch Cell Output:\n{h_t_torch.squeeze()}")
    print(f"Max Absolute Difference: {diff:.8f}")

    assert diff < 1e-5, "Test Failed: Kết quả giữa Custom Cell và PyTorch Cell không khớp!"
    print(">>> SUCCESS: Custom RNN Cell hoạt động hoàn toàn chính xác!")

if __name__ == "__main__":
    test_rnn_cell_implementation()