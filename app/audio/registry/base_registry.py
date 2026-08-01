"""Generic thread-safe Registry pattern base specification.

Provides the generic BaseRegistry[T] class for registering, instantiating,
and querying pluggable framework components.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Generic, List, Type, TypeVar

from app.audio.exceptions.model_exceptions import (
    ComponentNotFoundError,
    DuplicateRegistrationError,
)

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """Thread-safe generic component registry implementing the Registry Pattern.

    Args:
        name (str): Identifier name for the registry instance.
    """

    def __init__(self, name: str = "ComponentRegistry") -> None:
        self._name = name
        self._registry: Dict[str, Type[T] | Callable[..., T]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get registry identifier name."""
        return self._name

    def register(
        self, name: str, component: Type[T] | Callable[..., T], overwrite: bool = False
    ) -> None:
        """Register a component class or factory under a unique identifier.

        Args:
            name (str): Key name for component lookup.
            component (Type[T] | Callable[..., T]): Target class or factory callable.
            overwrite (bool): Whether to allow replacing an existing registration.

        Raises:
            DuplicateRegistrationError: If key name is already registered and overwrite is False.
        """
        key = name.lower().strip()
        with self._lock:
            if key in self._registry and not overwrite:
                raise DuplicateRegistrationError(
                    f"Component '{name}' is already registered in {self._name}."
                )
            self._registry[key] = component

    def get(self, name: str) -> Type[T] | Callable[..., T]:
        """Retrieve registered component class or factory by name.

        Args:
            name (str): Component key name.

        Returns:
            Type[T] | Callable[..., T]: Registered class or factory function.

        Raises:
            ComponentNotFoundError: If no component is registered under key name.
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
        """List all currently registered component names.

        Returns:
            List[str]: List of sorted registered component keys.
        """
        with self._lock:
            return sorted(list(self._registry.keys()))

    def unregister(self, name: str) -> None:
        """Remove a registered component from the registry.

        Args:
            name (str): Key name to remove.
        """
        key = name.lower().strip()
        with self._lock:
            if key in self._registry:
                del self._registry[key]
