# Attribution and dependency boundary

The target is Qwen/Qwen2.5-7B-Instruct, released by the Qwen team under Apache-2.0.
The AV and AR are kitft/nla-qwen2.5-7b-L20-av and -ar, distributed under Apache-2.0.
They are external pretrained models; this project did not train them.

The adapter follows the author-maintained
[inference recipe](https://github.com/kitft/nla-inference/tree/38b802a33d1d317f21b6825a9116f388c2141f86).
This repository does not vendor that research repository. Five small original
config/metadata files in `tests/fixtures/upstream/` are copied without modification
from the pinned model revisions in `configs/model-lock.json`; their hashes are
tested against the lock. The corresponding [Apache-2.0 license](third_party/APACHE-2.0.txt)
is included. New adapter code implements a documented local Transformers
alternative rather than importing upstream's SGLang client or enabling remote code.

Primary NLA attribution: Kit Fraser-Taliente, Subhash Kantamneni, Euan Ong and
coauthors, *Natural Language Autoencoders Produce Unsupervised Explanations of LLM
Activations* (2026), [paper](https://transformer-circuits.pub/2026/nla/).
Model cards also attribute training data derived from WildChat-1M (ODC-BY) and
Ultra-FineWeb (Apache-2.0), itself derived from FineWeb (ODC-BY). This task neither
downloads nor redistributes those datasets.

The root project package has no Phase Two dependency. This subproject pins direct
dependencies in `pyproject.toml`, and versions/artifact hashes for their resolved
dependencies in `uv.lock`. PyTorch comes from its CUDA 13 wheel index, explicitly
scoped to that package. SGLang, FlashAttention, vLLM and private application code
are not dependencies. Do not modify the root environment to satisfy this one.

All model weights are separately fetched into ignored local directories. A
hash establishes which artifact was used, not who executed a remote run.
