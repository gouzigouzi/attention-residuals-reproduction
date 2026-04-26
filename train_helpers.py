import itertools


def batch_token_chunks(chunks, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []


def shard_items_for_rank(items, rank, world_size):
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    return itertools.islice(items, rank, None, world_size)


def reorder_stream_for_training(ds, rank, world_size, seed, shuffle_buffer_size=10_000):
    if hasattr(ds, "shard"):
        try:
            ds = ds.shard(num_shards=world_size, index=rank)
        except TypeError:
            ds = ds.shard(world_size, rank)
    else:
        ds = shard_items_for_rank(ds, rank, world_size)

    if hasattr(ds, "shuffle"):
        try:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        except TypeError:
            ds = ds.shuffle(seed=seed)

    return ds


def split_token_buffer_into_chunks(buffer, seq_len):
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    chunk_size = seq_len
    chunks = []
    cursor = 0
    while len(buffer) - cursor >= chunk_size:
        chunks.append(buffer[cursor:cursor + chunk_size])
        cursor += chunk_size

    return chunks, buffer[cursor:]
