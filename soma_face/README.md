# face_api 

a microservice for face dection and finding face vectors (latents) using onnruntime ;] <br>

# building 
## locally 
we will assume you already have rust installed , if not , please install rust toolchain by visiting this link
[here](https://www.rust-lang.org/tools/install)
### 0. installing pre-requisites
#### installing onnxruntime
we will be using `onnxruntime`. please download and extract onnxruntime prebuilt binaries from [here](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.1) for your specific operating system. we will be assuming that you are using linux so we will be installing the linux binaries without any hardware acceleration. <br>
- download and extract `onnxruntime-linux-x64-1.18.1.tgz`
- in your `~/.zshrc` or `~/.bashrc` etc. , add the following line 
  ```bash
  export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-gpu-1.18.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  ```
- run `exec $SHELL` or `source ~/.zshrc` to refresh your env

#### installing required tools (apt)
```bash
sudo apt install -y cmake git build-essential wget
```

### 1. local development
running a debug build 
```bash
cargo run
```

building a *optimized* release build 
```bash
cargo build --release
```

### 2. docker
to build a local docker image, <br>
we assume that you have a registry running at `localhost:5000`

```bash
make local_container
```
you can also change the container tag/name by passing `LOCAL_CONTAINER_NAME`

```bash
make local_container LOCAL_CONTAINER_NAME=localhost:4000/face_api:0.4.1.2
```

to run locally in docker 

```bash
make run_local 
```

mapping to a different port by passing `LOCAL_SERVER_PORT`
```bash
make run_local LOCAL_SERVER_PORT=3930
```
