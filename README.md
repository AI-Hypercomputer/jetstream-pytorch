# petstream
JetStream Engine implementation in PyTorch


# Install torch_xla2

```bash
git clone https://github.com/pytorch/xla.git
cd xla/experimental/torch_xla2
pip install -e .
```

# Local run
```
python -m petstream.jet_engine_python_run --bf16_enable=True --context_length=8 --batch_size=2
```

# Bring up server
```
python -m run_server
By default it runs on 'tpu=4', add --platform='cpu=1' if you are running on CPU

Firing the request with:
python jetstream/core/tools/requester.py
```

# Profiling
```
export profiling_output = Some gcs bucket
python -m petstream.jet_engine_python_run --bf16_enable=True --context_length=8 --batch_size=2 --profiling_output={{profiling_output}}

Switch to your Cloud top, run:
export profiling_result = Some google generated folder in your gcs bucket
petstream/gcs_to_cns.sh {{profiling_result}}

The dump will always be in this directory: /cns/pi-d/home/{USER}/tensorboard/multislice/, load to Xprof/Offeline/Xplane
```
