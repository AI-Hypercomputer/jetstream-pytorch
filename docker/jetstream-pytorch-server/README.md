## Build and upload JetStream PyTorch Server image

These instructions are to build the JetStream PyTorch Server image, which calls an entrypoint script that invokes the [JetStream](https://github.com/AI-Hypercomputer/JetStream) inference server with the JetStream-PyTorch framework. 

```
docker build -t jetstream-pytorch-server .
docker tag jetstream-pytorch-server us-docker.pkg.dev/${PROJECT_ID}/jetstream/jetstream-pytorch-server:latest
docker push us-docker.pkg.dev/${PROJECT_ID}/jetstream/jetstream-pytorch-server:latest
```

If you would like to change the version of MaxText the image is built off of, change the `PYTORCH_JETSTREAM_VERSION` environment variable:
```
ENV PYTORCH_JETSTREAM_VERSION=<your desired commit hash, release, or tag>
```