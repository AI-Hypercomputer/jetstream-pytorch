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
python -m petstream.pets.jet_engine_python_run --bf16_enable=True --context_length=8 --batch_size=2
```
