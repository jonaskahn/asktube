import hashlib


def sha256(text: str) -> str:
    h = hashlib.new("sha256")
    h.update(text.encode("utf-8"))
    return h.hexdigest()
