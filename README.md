# Federated Learning dựa trên Blockchain với cơ chế điều chỉnh trọng số và bảo mật ZKP

[cite_start]Dự án nghiên cứu và xây dựng kiến trúc học liên kết (Federated Learning) an toàn, phi tập trung, tích hợp công nghệ Blockchain và bằng chứng không tiết lộ (Zero-Knowledge Proof) nhằm bảo vệ quyền riêng tư và chống tấn công độc hại[cite: 5, 6, 79].

---

## 📌 Tổng quan đề tài
[cite_start]Trong kỷ nguyên chuyển đổi số, việc huấn luyện mô hình trên dữ liệu lớn vấp phải rào cản về bảo mật dữ liệu cá nhân[cite: 13, 14]. Hệ thống này giải quyết các lỗ hổng của FL truyền thống như:
* [cite_start]**Nguy cơ tấn công:** Poisoning attacks và backdoor[cite: 16].
* [cite_start]**Thiếu minh bạch:** Sự phụ thuộc vào một máy chủ trung tâm không đáng tin cậy[cite: 16].
* [cite_start]**Kiểm soát chất lượng:** Khó kiểm soát tính trung thực của các bản cập nhật từ Client[cite: 16].

## 🏗 Kiến trúc hệ thống (Proposed Architecture)
[cite_start]Hệ thống được chia thành 4 lớp cốt lõi[cite: 53]:

1. **Client (Local Training Layer):**
   * [cite_start]Huấn luyện mô hình CNN cục bộ trên tập dữ liệu MNIST (phân phối Non-IID)[cite: 99, 100].
   * [cite_start]Sử dụng **ZK Prover** để tạo bằng chứng ($\pi$) xác nhận quá trình huấn luyện đúng thuật toán[cite: 28, 31, 58].
2. **Decentralized Storage Layer (IPFS):**
   * [cite_start]Lưu trữ các tệp trọng số mô hình có kích thước lớn để giảm tải cho Blockchain[cite: 64, 89].
   * [cite_start]Trả về mã băm định danh nội dung (CID) để tham chiếu[cite: 65].
3. **Blockchain (Consensus & Verification Layer):**
   * [cite_start]Sử dụng mạng **Ethereum Private** với cơ chế đồng thuận **Proof of Authority (PoA)**[cite: 60, 119].
   * [cite_start]**Smart Contract** đóng vai trò là bộ xác thực ZK Verifier và thực thi cơ chế thưởng - phạt[cite: 40, 41, 63].
4. **Aggregation Layer:**
   * [cite_start]Tổng hợp các bản cập nhật đã qua xác thực để tạo mô hình toàn cục (Global Model)[cite: 46, 52].

## ⚙️ Các kỹ thuật then chốt
* [cite_start]**Zero-Knowledge Proof (ZKP):** Chứng minh tính hợp lệ của cập nhật mà không làm lộ tham số nhạy cảm hoặc dữ liệu gốc[cite: 58, 85].
* [cite_start]**Adaptive Weight Adjustment:** Điều chỉnh trọng số thích nghi dựa trên chuẩn $L_2$ (Euclidean norm) để giảm ảnh hưởng của các cập nhật bất thường[cite: 69, 71, 121]:
  [cite_start]$$\gamma_{i}=exp(-\lambda\cdot\Delta_{i})$$ [cite: 70]
* [cite_start]**Federated Optimization:** Kết hợp **FedProx** tại Client để hạn chế *local drift* và **FedAdam** tại Server để tối ưu tốc độ hội tụ[cite: 72, 105, 107].
* [cite_start]**Reputation-Reward:** Quản lý uy tín Client dựa trên mức độ đóng góp thực tế qua các vòng huấn luyện[cite: 66, 87].
