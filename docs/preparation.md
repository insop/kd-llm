# env preparation

## docker preparation

see [docker_install.md](docker_install.md) for detail

## checkout git repos

see [git_repos.md](git_repos.md) for detail

## run this command in the docker

- download the teacher and student models

```bash
tune download meta-llama/Llama-3.2-1B --output-dir /workspace/.cache/tune/Meta-Llama-3.2-1B --ignore-patterns "original/consolidated.00.pth"

tune download meta-llama/Llama-3.2-3B --output-dir /workspace/.cache/tune/Meta-Llama-3.2-3B --ignore-patterns "original/consolidated.00.pth"

tune download meta-llama/Llama-3.1-8B --output-dir /workspace/.cache/tune/Meta-Llama-3.1-8B --ignore-patterns "original/consolidated.00.pth"
```
