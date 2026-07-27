import asyncio

# One batch GPU subprocess at a time, across the diarize endpoint and the job
# runner: their per-process allocator caps are sized assuming no other batch
# worker is resident.
batch_gpu_lock = asyncio.Lock()
