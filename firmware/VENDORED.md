# Vendored files — provenance and license

Parts of this repository are vendored from [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)
(MIT License, Copyright (c) 2026 Viacheslav Sierbov). Thank you — it's a
remarkable piece of engineering.

| File in this repo | Upstream location | Modifications |
|---|---|---|
| `firmware/common/llm.h` | `firmware/common/llm.h` | none (verbatim) |
| `firmware/md_expert/md_expert.ino` | (derivative of `esp32_llm` sketch) | replaced display path with serial GEN protocol |
| `firmware/tools/model.py` | `src/model.py` | none (verbatim) |
| `firmware/tools/quantize.py` | `src/quantize.py` | none (verbatim) |
| `firmware/tools/export.py` | `src/export.py` | none (verbatim) |
| `firmware/tools/train.py` | `src/train.py` | none (verbatim) |
| `firmware/tools/prepare_data.py` | `data/prepare.py` | none (verbatim) |
| `firmware/tools/gen_assets.py` | `src/gen_assets.py` | none (verbatim) |

All are MIT-licensed upstream; this repository remains MIT. If you reuse those
files standalone, credit Viacheslav Sierbov (slvDev).