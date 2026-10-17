"""Native camera latents reach host export and optional GPU preview readers."""

import threading
import time
from contextlib import ExitStack

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve_worker.protocol.transfer import TensorTransfer, WorkerEndpoint
from uniserve_worker.transport import make_transports
from uniserve_worker.transport.fetch import fetch_tensor
from uniserve_worker.transport.publication import publish_tensor

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("include_preview", (False, True))
@torch.inference_mode()
def test_native_camera_host_product_reaches_all_consumers(include_preview):
    """VMM registration must not prevent a host product's CPU/GPU delivery.

    The final camera artifact and its preview read the same native latent.
    Use distinct real SHM endpoints with independent acknowledgment slots,
    while the producer also binds VMM as it does in a multi-GPU deployment.
    """
    shape = (1, 16, 8, 2, 4)
    expected = torch.arange(1024, dtype=torch.float32).reshape(shape).bfloat16()
    source = expected.to("cuda:0")
    consumers = (1, 2) if include_preview else (1,)
    with ExitStack() as scope:
        events = EventPool()
        scope.callback(events.close)
        producers = make_transports(
            ("cuda_vmm", "posix_shm"),
            byte_capacity=16384,
            ticket_capacity=2,
            event_pool=events,
            host_slots=consumers,
        )
        for transport in producers.values():
            scope.callback(transport.close)
        receivers = []
        for slot in consumers:
            receiver_events = EventPool()
            scope.callback(receiver_events.close)
            receiver = make_transports(
                ("posix_shm",),
                byte_capacity=16384,
                ticket_capacity=1,
                event_pool=receiver_events,
                source=WorkerEndpoint.local(),
                acknowledgment_slot=slot,
            )["posix_shm"]
            scope.callback(receiver.close)
            receivers.append(receiver)

        retirements = []
        locations = publish_tensor(
            producers,
            source,
            retain=retirements.append,
            consumers=consumers,
            host=True,
        )
        for location in locations:
            scope.callback(producers[location.backend].release, location)
        product = TensorTransfer(shape=shape, locations=locations)
        destinations = [torch.empty_like(expected)]
        if include_preview:
            destinations.append(torch.empty_like(source))
        for receiver, destination in zip(receivers, destinations, strict=True):
            tickets = fetch_tensor(
                product,
                destination,
                bindings={
                    (location.source, "posix_shm"): receiver
                    for location in locations
                },
            )
            for ticket in tickets:
                scope.callback(ticket.close)
                ready = threading.Event()
                ticket.add_done_callback(ready.set)
                assert ready.wait(30), "camera latent read did not complete"
                ticket.result()
                ticket.close()
            assert destination.shape == expected.shape
            assert destination.dtype == expected.dtype
            assert torch.equal(
                destination.cpu().view(torch.uint8), expected.view(torch.uint8)
            )

        # The same bounded publication capacity must be reusable after both
        # export and preview have acknowledged their read, without leaking a
        # source reservation when the camera result leaves the request graph.
        for location in locations:
            producers[location.backend].release(location)
        deadline = time.monotonic() + 30
        while not all(future.done() for future in retirements):
            for transport in producers.values():
                transport.reap()
            assert time.monotonic() < deadline, "camera product did not retire"
            time.sleep(0.001)
        for future in retirements:
            future.result()
