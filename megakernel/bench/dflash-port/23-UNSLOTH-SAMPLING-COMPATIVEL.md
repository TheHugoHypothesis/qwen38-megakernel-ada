# 23 — Unsloth Studio: compatible sampling options

2026-10-11. A user reported `presence_penalty is not supported by the megakernel (invalid_request_error)` when connecting Studio to the MK server.

The local server rejected a request field that the client sends by default. The HTTP adapter was updated to accept and apply `presence_penalty` through the supported sampling path. The fix keeps unsupported-option errors explicit and avoids silently ignoring fields. The associated request checks and code paths are preserved with the server source.
