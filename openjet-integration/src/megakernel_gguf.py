"""Read local GGUF metadata without numpy; CUDA loader validates the full schema."""
import mmap
import struct
from pathlib import Path


def compatible_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            pos = 0

            def read(fmt):
                nonlocal pos
                n = struct.calcsize(fmt)
                if pos + n > len(data):
                    raise ValueError("truncated GGUF")
                value = struct.unpack_from("<" + fmt, data, pos)[0]
                pos += n
                return value

            def string(keep=True):
                nonlocal pos
                n = read("Q")
                if n > len(data) - pos:
                    raise ValueError("truncated string")
                value = bytes(data[pos:pos+n]).decode("utf-8") if keep else None
                pos += n
                return value

            formats = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}

            def value(kind, keep):
                nonlocal pos
                if kind == 8:
                    return string(keep)
                if kind == 9:
                    item, count = read("I"), read("Q")
                    if count > len(data) or item == 9:
                        raise ValueError("invalid array")
                    if not keep and item in formats:
                        pos += count * struct.calcsize(formats[item])
                        if pos > len(data):
                            raise ValueError("truncated array")
                        return None
                    if keep:
                        return [value(item, True) for _ in range(count)]
                    for _ in range(count):
                        value(item, False)
                    return None
                return read(formats[kind])

            if read("I") != 0x46554747 or read("I") not in (2, 3):
                return False
            tensors, fields = read("Q"), read("Q")
            if tensors not in (851, 866) or fields > 1_000_000:
                return False
            meta = {}
            for _ in range(fields):
                key = string()
                kind = read("I")
                keep = key.startswith("qwen35.") or key == "general.architecture"
                v = value(kind, keep)
                if keep:
                    meta[key] = v
            expected = {"general.architecture":"qwen35", "qwen35.block_count":64+meta.get("qwen35.nextn_predict_layers",0),
                        "qwen35.embedding_length":5120,"qwen35.feed_forward_length":17408,
                        "qwen35.attention.head_count":24,"qwen35.attention.head_count_kv":4,
                        "qwen35.ssm.state_size":128,
                        "qwen35.rope.dimension_sections":[11,11,10,0]}
            if meta.get("qwen35.nextn_predict_layers",0) not in (0,1):
                return False
            if any(meta.get(k) != v for k,v in expected.items()):
                return False
            seen = {}
            supported = {0, 1, 8, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 29, 30}
            for _ in range(tensors):
                name, rank = string(), read("I")
                if not 1 <= rank <= 4 or name in seen:
                    return False
                dims = tuple(read("Q") for _ in range(rank))
                kind, offset = read("I"), read("Q")
                if kind not in supported or not all(dims) or offset > len(data):
                    return False
                seen[name] = dims, kind
            emb = seen.get("token_embd.weight", (None,None))[0]
            if not emb or len(emb)!=2 or emb[0]!=5120 or not 32<=emb[1]<=248320 or emb[1]%2:
                return False
            mtp = meta.get("qwen35.nextn_predict_layers",0)
            return (seen.get("output.weight", (None,None))[0] == emb
                    and seen["output.weight"][1] in {8,12,13,14}
                    and (seen.get("blk.64.nextn.eh_proj.weight",(None,None))[0] == (10240,5120) if mtp else not any(n.startswith("blk.64.") for n in seen)))
    except (OSError, ValueError, KeyError, struct.error, UnicodeError, OverflowError):
        return False
