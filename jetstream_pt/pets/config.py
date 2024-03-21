from jetstream.core.config_lib import ServerConfig
from petstream.jet_engine2 import create_pytorch_engine


def create_config(
            devices,
            tokenizer_path,
            ckpt_path,
            bf16_enable,
            param_size,
            context_length,
            batch_size,
            platform,
):
    def func(a):
        return create_pytorch_engine(
            devices=devices,
            tokenizer_path=tokenizer_path,
            ckpt_path=ckpt_path,
            bf16_enable=bf16_enable,
            param_size=param_size,
            context_length=context_length,
            batch_size=batch_size,
        )


    return ServerConfig(
        interleaved_slices=(platform, ),
        interleaved_engine_create_fns=(func, ),
    )
