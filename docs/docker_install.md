
# Docker install


## Use this document to install docker 
- https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04

Then last step with this instead of `su` which requires password.
```bash
sudo -i -u ${USER}
```

## install nvidia container toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-yum-or-dnf

If you don't you may see this error
```bash
$ docker run --gpus all 112a05617efe
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
