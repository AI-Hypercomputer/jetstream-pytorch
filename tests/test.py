import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu

P = jax.sharding.PartitionSpec

num_devices = jax.local_device_count()
assert num_devices > 1, "Please run this notebook with more than one device."
assert "TPU" in jax.devices()[0].device_kind, "Please run this notebook with TPU devices."
print(f"Running with {num_devices} {jax.devices()[0].device_kind} devices.")

partition = P('x', None)
devices = mesh_utils.create_device_mesh((num_devices, 1))
mesh = jax.sharding.Mesh(devices, partition)
sharding = jax.sharding.NamedSharding(mesh, partition)

# Create an input array that shards the first dimension across
# all devices.
input_arr = jax.random.uniform(jax.random.key(0), (8 * num_devices, 128))
input_arr = jax.device_put(input_arr, sharding)


def all_gather_kernel(input_ref,
                      output_ref,
                      local_copy_sem,
                      send_sem,
                      recv_sems):
  outer_step = pl.program_id(0)
  my_id = lax.axis_index('x')
  jax.debug.print("outer_step: {}, my_id: {}", outer_step, my_id)
  # jax.debug.print("input_ref: {}, output_ref: {}", input_ref, output_ref)
  right_neighbor = lax.rem(my_id + 1, num_devices)
  copy_slot = my_id - outer_step
  copy_slot = lax.rem(copy_slot + num_devices, num_devices)

  @pl.when(outer_step == 0)
  def _():
    local_copy_op = pltpu.make_async_copy(
      src_ref=input_ref,
      dst_ref=output_ref.at[my_id],
      sem=local_copy_sem,
    )
    local_copy_op.start()
    local_copy_op.wait()

  # Copy to our right neighbor.
  # Note that we will also be receiving data from our left neighbor,
  # but at `copy_slot-1` rather than `copy_slot`! This makes use of the fact
  # that the indices do not need to be symmetric between remote DMAs.
  remote_copy_op = pltpu.make_async_remote_copy(
      src_ref=output_ref.at[copy_slot],
      dst_ref=output_ref.at[copy_slot],
      send_sem=send_sem,
      recv_sem=recv_sems.at[outer_step],
      device_id=(right_neighbor, 0),
      device_id_type=pltpu.DeviceIdType.MESH,
  )
  remote_copy_op.start()
  remote_copy_op.wait()

out_shape = jax.ShapeDtypeStruct((num_devices, 8, 128), jnp.float32)
grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                # TPUMemorySpace.ANY will (usually) place the tensor in HBM.
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            scratch_shapes=(
              # DMA semaphores are allocated in scratch memory.
              # We allocated one semaphore for a local HBM-VMEM copy,
              # and one for the remote send semaphore.
              [pltpu.SemaphoreType.DMA] * 2
              # We additionally allocate one receive semaphore per device.
              # This is to avoid situations where we have multiple
              # DMAs in flight, as we do not want to share a receive
              # semaphore between the DMAs.
              + [pltpu.SemaphoreType.DMA((num_devices-1,))]

            ),
            grid=(num_devices-1,)
        )

all_gather = pl.pallas_call(
      all_gather_kernel,
      out_shape=out_shape,
      grid_spec=grid_spec,
      interpret=True,
  )

# Wrap the kernel within a shard_map to call.
# pallas_result = jax.jit(
#       shard_map.shard_map(
#           all_gather,
#           mesh=mesh,
#           in_specs=partition,
#           out_specs=partition,
#           check_rep=False
#       )
# )(input_arr)

# Wrap the kernel within a shard_map to call.
pallas_result = shard_map.shard_map(
          all_gather,
          mesh=mesh,
          in_specs=partition,
          out_specs=partition,
          check_rep=False
      )(input_arr)

# Compare Pallas result to XLA shard_map result.
xla_result = jax.jit(
    shard_map.shard_map(
        lambda x: lax.all_gather(x, 'x'),
        mesh=mesh, in_specs=partition, out_specs=partition
    )
)(input_arr)

print('Input: ', input_arr.shape, input_arr[::8, 0])
print('Pallas Result: ', pallas_result.shape, pallas_result[:, 0, 0])
print('lax.all_gather Result: ', xla_result.shape, xla_result[:, 0, 0])
print('Difference |Pallas - lax.all_gather| = ',
      jnp.mean(jnp.abs(pallas_result - xla_result)))