from web3 import Web3
import json
from reputation import update_reputation

# Kết nối Ganache
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
contract_address = "0x73757314f1CAf88110E194e3243cd0eeD2Fb24eB"

with open("build/contracts/Reputation.json", "r", encoding="utf-8") as f:
    contract_json = json.load(f)

abi = contract_json["abi"]
contract = w3.eth.contract(address=contract_address, abi=abi)
account = w3.eth.accounts[0]

# ==========================================
# Submit client update
# ==========================================
def submit_update(round_num, client_id, cid, proof_hash, accuracy):
    # Đảm bảo ép kiểu về int cho các trường uint256 trong Solidity
    tx = contract.functions.submitUpdate(
        int(round_num),
        int(client_id),
        str(cid), # cid này có thể là tên hoặc hash (string)
        str(proof_hash),
        int(accuracy * 1000)
    ).transact({"from": account})

    receipt = w3.eth.wait_for_transaction_receipt(tx)
    return receipt.transactionHash.hex()

# ==========================================
# Verify update + update reputation (SỬA LỖI TẠI ĐÂY)
# ==========================================
def verify_update(client_id, round_num, result):
    try:
        # ÉP KIỂU: Chuyển "4" (str) thành 4 (int) để khớp với uint256
        tx = contract.functions.verifyUpdate(
            int(client_id),   # Sửa lỗi Argument 1
            int(round_num),   # Sửa lỗi Argument 2
            bool(result)      # Đảm bảo là Boolean
        ).transact({"from": account})

        w3.eth.wait_for_transaction_receipt(tx)

        # Cập nhật reputation trong file local (JSON)
        # Nếu result là True, ta truyền điểm 1.0 (hoặc tùy logic của bạn)
        score = 1.0 if result else -1.0
        rep = update_reputation(client_id, score)

        print(f"✅ Blockchain Verify → Client {client_id} | Round {round_num} | Rep: {rep:.3f}")
        return True
    except Exception as e:
        print(f"❌ Blockchain Error: {e}")
        return False

# ==========================================
# Query reputation
# ==========================================
def get_reputation(client_id):
    # Ép kiểu int khi gọi hàm call()
    return contract.functions.getReputation(int(client_id)).call()