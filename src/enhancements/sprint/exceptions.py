"""
Custom exceptions for Sprint module.

This module defines specific exception types for Sprint-related errors
to provide better error handling and user feedback.
"""

from typing import Optional, Dict, Any


class SprintError(Exception):
    """Base exception for Sprint module errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class SprintConfigurationError(SprintError):
    """Raised when Sprint configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 config_value: Optional[Any] = None, expected: Optional[str] = None):
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = config_value
        if expected:
            details['expected'] = expected

        super().__init__(message, details)


class SprintImportError(SprintError):
    """Raised when Sprint dependencies are missing or import fails."""

    def __init__(self, message: str, missing_module: Optional[str] = None,
                 suggestion: Optional[str] = None):
        details = {}
        if missing_module:
            details['missing_module'] = missing_module
        if suggestion:
            details['suggestion'] = suggestion

        super().__init__(message, details)


class SprintCompatibilityError(SprintError):
    """Raised when Sprint is incompatible with current setup or configuration."""

    def __init__(self, message: str, incompatible_feature: Optional[str] = None,
                 alternative: Optional[str] = None):
        details = {}
        if incompatible_feature:
            details['incompatible_feature'] = incompatible_feature
        if alternative:
            details['alternative'] = alternative

        super().__init__(message, details)


class SprintModelStateError(SprintError):
    """Raised when Sprint model state is inconsistent or corrupted."""

    def __init__(self, message: str, state_check: Optional[str] = None,
                 recovery_action: Optional[str] = None):
        details = {}
        if state_check:
            details['state_check'] = state_check
        if recovery_action:
            details['recovery_action'] = recovery_action

        super().__init__(message, details)


class SprintDeviceError(SprintError):
    """Raised when Sprint encounters device-related issues."""

    def __init__(self, message: str, device: Optional[str] = None,
                 tensor_name: Optional[str] = None):
        details = {}
        if device:
            details['device'] = device
        if tensor_name:
            details['tensor_name'] = tensor_name

        super().__init__(message, details)


class SprintMemoryError(SprintError):
    """Raised when Sprint encounters memory-related issues."""

    def __init__(self, message: str, required_mb: Optional[int] = None,
                 available_mb: Optional[int] = None):
        details = {}
        if required_mb:
            details['required_mb'] = required_mb
        if available_mb:
            details['available_mb'] = available_mb

        super().__init__(message, details)