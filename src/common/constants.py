"""
Project-wide constants and configuration values.

This module contains constants that are used across multiple parts of the codebase
and should be centralized for maintainability.
"""

# WAN Architecture Constants
# =========================

# Resolution step size for WAN 2.x architecture
# This determines the granularity of resolution bucketing for multi-resolution training
# WAN 2.x requires resolutions to be divisible by this value for optimal performance
RESOLUTION_STEPS_WAN_2 = 16
