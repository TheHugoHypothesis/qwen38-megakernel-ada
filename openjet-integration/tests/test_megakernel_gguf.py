"""Eligibility must follow GGUF metadata rather than the download's filename."""
from pathlib import Path
import tempfile
from src.megakernel_gguf import compatible_gguf
from src import llama_server


def test_corrupt_gguf_with_supported_filename_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/"Qwen3.8-27B-Q4_K_M.gguf"
        p.write_bytes(b"GGUF")
        assert not compatible_gguf(p)
        assert not llama_server._megakernel_model_supported(str(p))


def test_nonexistent_original_download_keeps_build_hint():
    assert llama_server._megakernel_model_supported("/missing/Qwen3.8-27B-Q4_K_M.gguf")
    assert not llama_server._megakernel_model_supported("/missing/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")


def test_actual_gsq_model_metadata_if_available():
    import pytest
    candidates=list(Path(__file__).resolve().parents[2].glob("models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF/snapshots/*/*mtp.gguf"))
    if not candidates:
        pytest.skip("local GSQ-RCO fixture unavailable")
    assert compatible_gguf(candidates[0])
    assert llama_server._megakernel_model_supported(str(candidates[0]))
