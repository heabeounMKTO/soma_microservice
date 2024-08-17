FROM rust:latest as builder
COPY soma_core .
WORKDIR /home
ENV DEBIAN_FRONTEND=noninteractive
COPY ./src ./src
COPY ./Cargo.toml ./Cargo.toml
RUN apt-get update && apt-get upgrade -y 
RUN apt install -y wget  
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
RUN tar -xvf onnxruntime-linux-x64-1.18.1.tgz
ENV LD_LIBRARY_PATH=/home/onnxruntime-linux-x64-1.18.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
RUN cargo build --release

FROM debian:sid-slim as runtime 
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home 
COPY --from=builder /home/target/release/soma_face /home
COPY ./models ./models
COPY .env .
ENTRYPOINT [ "/home/soma_face" , "--workers" , "4"]

