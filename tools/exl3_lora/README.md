# GLM-5.2 EXL3 v31 source baseline

This branch starts from the exact vLLM source installed in the published
GLM-5.2 EXL3 v31 runtime.

| Artifact | Identity |
| --- | --- |
| Base commit | `0c79e41db41f250ccdfc4be92d171960a5787f73` |
| Runtime image | `verdictai/glm52-exl3-sparkinfer:v31-gg-v20-sic3828fd-vllm0c79e41-cu132-sm120a` |
| Runtime index digest | `sha256:0433ae94665b769b78dd301f952d907508a3ba80bce47a1630ec20ade8812dff` |
| Linux/amd64 manifest | `sha256:90bbc355c0201445990ebcda8a1e7a302dd165c321b534eb21639c1c197703a1` |
| vLLM overlay layer | `sha256:ce24295c898c460e4579a0447ca487f69b5bc62b6d10b6c8337a602165f646ac` |
| Overlay manifest digest | `281718461404120a7ea2ef2725166dcf11c4aa29b83c0183f6998bdd91caf60c` |

`v31-overlay.sha256` lists every source file copied by the image's
`COPY overlay/vllm/ .../site-packages/vllm/` layer. To verify a checkout:

```bash
shasum -a 256 -c tools/exl3_lora/v31-overlay.sha256
shasum -a 256 tools/exl3_lora/v31-overlay.sha256
```

The second command hashes the manifest file bytes. Its result is the overlay
manifest digest in the table above.
