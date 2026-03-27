# Blockchain-based Federated Learning with ZKP

## Giới thiệu

Đây là đồ án chuyên ngành xây dựng một hệ thống Federated Learning (FL) an toàn.

Tích hợp:
- Blockchain (Ethereum Private – PoA)
- Federated Learning (FedProx + FedAdam)
- IPFS
- Zero-Knowledge Proof (ZKP)
- Reputation – Reward

Mục tiêu là xác thực và kiểm soát cập nhật mô hình trong môi trường phân tán.

## Mục tiêu

- Xây dựng hệ thống FL an toàn
- Ngăn chặn update độc hại
- Bảo vệ dữ liệu client
- Tối ưu hội tụ Non-IID

## Kiến trúc

Client → IPFS → Blockchain → Server

## Công nghệ

- Flower
- PyTorch
- Ethereum
- IPFS
- Solidity
