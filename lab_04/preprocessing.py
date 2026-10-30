
import re

def tokenize_review(text):
    """
    Chuyển bài đánh giá thành danh sách token
    để sử dụng với Word2Vec.
    """
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
