import hashlib
import numpy as np
import faiss


def stable64(s: str) -> np.int64:
    # prefer faiss.hash64 if available
    if hasattr(faiss, "hash64"):
        return np.int64(faiss.hash64(s))
    # portable fallback: first 8 bytes of BLAKE2b
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(h, dtype="<i8")[0]  # <i8 = little-endian int64
