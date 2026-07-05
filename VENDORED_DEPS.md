# Vendored dependencies

The following third-party repositories are vendored as regular source
directories so this project can be synced as a single Git repository.

| Directory | Upstream | Commit |
| --- | --- | --- |
| `BenchMARL/` | `https://github.com/facebookresearch/BenchMARL.git` | `65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1` |
| `VectorizedMultiAgentSimulator/` | `https://github.com/proroklab/VectorizedMultiAgentSimulator.git` | `9658bc56e886f96e4b62033e5df009eccb92122b` |
| `marl-eval/` | `https://github.com/instadeepai/marl-eval.git` | `5675ad67e8eeb7450bbbe827883849a938b27d80` |

Local experiment patches from this repository are applied directly inside these
directories, including the CMASAC/Test BenchMARL algorithm files and the VMAS
`core.py` change described in `README.md`.
