"""Simple training event system for junction points."""

from typing import Any, Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Event handlers registry - just a simple dict
_event_handlers: Dict[str, List[Callable]] = {}

def register_handler(event_name: str, handler: Callable) -> None:
    """Register a handler for a training event."""
    if event_name not in _event_handlers:
        _event_handlers[event_name] = []
    _event_handlers[event_name].append(handler)

def trigger_event(event_name: str, **kwargs) -> Dict[str, Any]:
    """Trigger all handlers for an event, return aggregated results."""
    results = {}
    if event_name in _event_handlers:
        for handler in _event_handlers[event_name]:
            try:
                result = handler(**kwargs)
                if result:
                    results.update(result)
            except Exception as e:
                logger.warning(f"Event handler {handler.__name__} failed for {event_name}: {e}")
    return results