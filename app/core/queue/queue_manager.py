"""Thread-safe bounded queue and queue manager system for producer-consumer architecture."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar("T")


@dataclass
class QueueStats:
    """Dataclass holding queue performance and throughput metrics."""

    name: str
    size: int
    maxsize: int
    total_enqueued: int
    total_dequeued: int
    total_dropped: int

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return self.size >= self.maxsize if self.maxsize > 0 else False

    @property
    def drop_rate(self) -> float:
        """Calculate drop percentage relative to total enqueued items."""
        total = self.total_enqueued + self.total_dropped
        return (self.total_dropped / total * 100.0) if total > 0 else 0.0


class BoundedQueue(Generic[T]):
    """Thread-safe bounded queue supporting drop-oldest overflow policy."""

    def __init__(self, name: str = "default", maxsize: int = 128, drop_oldest: bool = True) -> None:
        self.name = name
        self.maxsize = maxsize
        self.drop_oldest = drop_oldest
        self._deque: collections.deque[T] = collections.deque(maxlen=maxsize if drop_oldest else None)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._total_dropped: int = 0

    def put(self, item: T, block: bool = False, timeout: Optional[float] = None) -> bool:
        """Enqueue item into bounded queue.

        Args:
            item: Item to enqueue.
            block: Whether to block if queue is full (only when drop_oldest is False).
            timeout: Maximum block wait time.

        Returns:
            bool: True if enqueued, False if dropped/full.
        """
        with self._lock:
            if not self.drop_oldest and self.maxsize > 0 and len(self._deque) >= self.maxsize:
                if not block:
                    return False
                end_time = time.monotonic() + (timeout or 0.0)
                while len(self._deque) >= self.maxsize:
                    remaining = end_time - time.monotonic()
                    if timeout is not None and remaining <= 0:
                        return False
                    self._not_empty.wait(timeout=remaining if timeout is not None else None)

            if self.drop_oldest and self.maxsize > 0 and len(self._deque) >= self.maxsize:
                self._deque.popleft()
                self._total_dropped += 1

            self._deque.append(item)
            self._total_enqueued += 1
            self._not_empty.notify()
            return True

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[T]:
        """Dequeue item from bounded queue."""
        with self._lock:
            if not self._deque:
                if not block:
                    return None
                end_time = time.monotonic() + (timeout or 0.0)
                while not self._deque:
                    remaining = end_time - time.monotonic()
                    if timeout is not None and remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining if timeout is not None else None)
            
            item = self._deque.popleft()
            self._total_dequeued += 1
            return item

    def get_stats(self) -> QueueStats:
        """Fetch snapshot of queue performance metrics."""
        with self._lock:
            return QueueStats(
                name=self.name,
                size=len(self._deque),
                maxsize=self.maxsize,
                total_enqueued=self._total_enqueued,
                total_dequeued=self._total_dequeued,
                total_dropped=self._total_dropped,
            )

    def clear(self) -> None:
        """Drain all elements from queue."""
        with self._lock:
            self._deque.clear()


class QueueManager:
    """Registry and orchestrator for named bounded queues across runtime pipeline."""

    def __init__(self) -> None:
        self._queues: Dict[str, BoundedQueue[Any]] = {}
        self._lock = threading.Lock()

    def create_queue(self, name: str, maxsize: int = 128, drop_oldest: bool = True) -> BoundedQueue[Any]:
        """Create and register a named BoundedQueue instance."""
        with self._lock:
            queue: BoundedQueue[Any] = BoundedQueue(name=name, maxsize=maxsize, drop_oldest=drop_oldest)
            self._queues[name] = queue
            return queue

    def get_queue(self, name: str) -> Optional[BoundedQueue[Any]]:
        """Retrieve existing named BoundedQueue instance."""
        with self._lock:
            return self._queues.get(name)

    def get_all_stats(self) -> Dict[str, QueueStats]:
        """Get statistics summary for all registered queues."""
        with self._lock:
            return {name: q.get_stats() for name, q in self._queues.items()}

    def clear_all(self) -> None:
        """Clear all registered queues."""
        with self._lock:
            for q in self._queues.values():
                q.clear()
