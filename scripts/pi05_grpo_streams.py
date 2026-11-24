"""Accumulate GRPO gradients with bounded lookahead on CUDA streams."""

from dataclasses import dataclass
from time import perf_counter

import torch

from pi05_rl import FlowTrace, recompute_flow


@dataclass
class Prepared:
    prefix: tuple
    trace: FlowTrace
    ready: torch.cuda.Event | None = None
    done: torch.cuda.Event | None = None


def host_batches(rows, size):
    result = []
    positions = torch.arange(rows[0][1]["trace"]["old_logprobs"].shape[1])
    for offset in range(0, len(rows), size):
        group = rows[offset:offset + size]
        count = len(group)
        # Pad the tail to the requested shape. Padding has zero loss weight.
        padded = group + [group[-1]] * (size - count)
        observations = [record["batch"] for _, record, _, _ in padded]
        obs = {key: torch.cat([batch[key] for batch in observations]).pin_memory()
               for key in observations[0] if key.startswith("observation.")}
        traces = [record["trace"] for _, record, _, _ in padded]
        trace = {
            key: torch.cat([item[key] for item in traces]).pin_memory() if torch.is_tensor(value) else value
            for key, value in traces[0].items()
        }
        advantage = torch.tensor([item[2] for item in padded]).pin_memory()
        weights = torch.tensor([item[3] for item in group] + [0.] * (size - count)).pin_memory()
        starts = torch.tensor([item[0].get("execution_start", 0) for item in padded])
        ends = starts + torch.tensor([item[0]["executed_steps"] for item in padded])
        mask = (positions >= starts[:, None]) & (positions < ends[:, None])
        mask[count:] = False
        result.append((obs, trace, advantage, weights, mask.pin_memory(), count))
    return result


def prepare(adapter, row):
    obs, recorded, *_ = row
    batch = {key: value.to("cuda", non_blocking=True) for key, value in obs.items()}
    trace = FlowTrace(**{key: value.to("cuda", non_blocking=True) if torch.is_tensor(value) else value
                         for key, value in recorded.items()})
    # Both variants consume a fresh prefix exactly once. The group reader checked
    # observation digests on CPU before the pinned copies were made.
    return Prepared(adapter.prefix(batch), trace)


def backward(adapter, prepared, row, totals):
    advantage, weight, mask = (x.to("cuda", non_blocking=True) for x in row[2:5])
    pad, cache = prepared.prefix
    trace = prepared.trace
    new = recompute_flow(
        lambda action, time: adapter.core.denoise_step(pad, cache, action, time),
        trace, adapter.cfg, policy_version=adapter.policy_version,
    )
    delta = (new - trace.old_logprobs) * mask[..., None]
    ratio = delta.sum(dim=(1, 2)).exp()
    clipped = ratio.clamp(1 - adapter.cfg.clip_low, 1 + adapter.cfg.clip_high)
    loss = -(torch.minimum(ratio * advantage, clipped * advantage) * weight).sum()
    loss.backward()
    with torch.no_grad():
        totals[0] += loss
        totals[1] = torch.maximum(totals[1], delta.abs().max())
        totals[2] = torch.maximum(totals[2], (ratio - 1).abs().max())
        valid = weight > 0
        totals[3] += ((ratio != clipped) & valid).sum()
        totals[4] += (ratio * valid).sum()


def accumulate(adapter, rows, prefix_streams):
    adapter.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = perf_counter()
    totals = torch.zeros(5, device="cuda")

    if not prefix_streams:
        for row in rows:
            backward(adapter, prepare(adapter, row), row, totals)
    else:
        backward_stream = torch.cuda.current_stream()
        lookahead = len(prefix_streams)
        slots = [None] * (lookahead + 1)

        def submit(index):
            slot = index % len(slots)
            previous = slots[slot]
            if previous is not None:
                # Keep ownership of prefix/cache storage until its consumer is done.
                previous.done.synchronize()
            prefix_stream = prefix_streams[index % lookahead]
            with torch.cuda.stream(prefix_stream):
                packet = prepare(adapter, rows[index])
                packet.ready = prefix_stream.record_event()
            slots[slot] = packet

        for index in range(min(lookahead, len(rows))):
            submit(index)
        for index, row in enumerate(rows):
            ahead = index + lookahead
            if ahead < len(rows):
                submit(ahead)
            packet = slots[index % len(slots)]
            backward_stream.wait_event(packet.ready)
            backward(adapter, packet, row, totals)
            packet.done = backward_stream.record_event()

    torch.cuda.synchronize()
    elapsed = perf_counter() - start_time
    memory = torch.cuda.max_memory_allocated() / 2**20
    if not torch.isfinite(totals).all():
        raise ValueError("Non-finite GRPO loss or probability ratio")
    loss, error, ratio_error, clipped, sum_ratio = totals.tolist()
    chunks = sum(row[-1] for row in rows)
    return {"streams": 1 + len(prefix_streams), "seconds": elapsed, "chunks": chunks,
            "chunks_per_second": chunks / elapsed, "peak_allocated_mib": memory,
            "loss": loss, "old_logprob_max_error": error, "old_ratio_max_error": ratio_error,
            "clip_fraction": clipped / chunks, "ratio_mean": sum_ratio / chunks}
