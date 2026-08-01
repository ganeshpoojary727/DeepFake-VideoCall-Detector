"""Thread-safe generic registry base class for Video AI components.

Provides generic BaseRegistry[T] for registering, retrieving, and listing
video subsystem components such as models, datasets, preprocessors, loss functions.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Generic, List, Type, TypeVar

from app.video.exceptions.video_exceptions import (
    ComponentNotFoundError,
    DuplicateRegistrationError,
)

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Thread-safe generic component registry pattern implementation.

    Args:
        name (str): Unique name identifier for the registry.
    """

    def __init__(self, name: str = "BaseRegistry") -> None:
        self._name: str = name
        self._registry: Dict[str, Type[T] | Callable[..., T]] = {}
        self._lock: threading.Lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get the registry instance name."""
        return self._name

    def register(
        self,
        name: str,
        component: Type[T] | Callable[..., T],
        overwrite: bool = False,
    ) -> None:
        """Register a component class or factory callable under a string key.

        Args:
            name: Component lookup key.
            component: Component class or factory function.
            overwrite: Whether to overwrite existing registration.

        Raises:
            DuplicateRegistrationError: If name already exists and overwrite is False.
        """
        key = name.lower().strip()
        with self._lock:
            if key in self._registry and not overwrite:
                raise DuplicateRegistrationError(
                    f"Component '{name}' is already registered in {self._name}."
                )
            self._registry[key] = component

    def get(self, name: str) -> Type[T] | Callable[..., T]:
        """Retrieve a registered component class or factory by key name.

        Args:
            name: Component lookup key.

        Returns:
            Registered class or factory callable.

        Raises:
            ComponentNotFoundError: If key name is not found in registry.
        """
        key = name.lower().strip()
        with self._lock:
            if key not in self._registry:
                available = ", ".join(sorted(self._registry.keys()))
                raise ComponentNotFoundError(
                    f"Component '{name}' not found in {self._name}. Available: [{available}]"
                )
            return self._registry[key]

    def list_registered(self) -> List[str]:
        """List all currently registered component lookup keys.

        Returns:
            Sorted list of registered key names.
        """
        with self._lock:
            return sorted(list(self._registry.keys()))

    def unregister(self, name: str) -> None:
        """Unregister a component key from the registry."""
        key = name.lower().strip()
        with self._lock:
            if key in self._registry:
                del self._registry[key]
