
## setup key

**SET USER AND EMAIL**

mkdir ~/$USER/.ssh
ssh-keygen -t ed25519 -C "$EMAIL" -f ~/$USER/.ssh/id_ed25519


chmod 600 ~/$USER/.ssh/id_ed25519
chmod 644 ~/$USER/.ssh/id_ed25519.pub


## Git repos

export GIT_SSH_COMMAND='ssh -i ~/$USER/.ssh/id_ed25519'

git clone https://github.com/$USER/torchtune
git clone https://github.com/pytorch/ao
git clone https://github.com/OpenCoder-llm/OpenCoder-llm.git
git clone https://github.com/evalplus/evalplus.git