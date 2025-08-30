"""Advanced logging configuration utilities."""

import argparse
import logging
from common.performance_logger import configure_verbosity

logger = logging.getLogger(__name__)


def configure_advanced_logging(args: argparse.Namespace) -> None:
    """Configure advanced logging settings including parameter stats, per-source losses, and gradient norms.

    Sets default values for various logging options if not already set in args.
    """
    # Performance logging verbosity
    if not hasattr(args, "performance_verbosity"):
        args.performance_verbosity = "standard"  # Default verbosity level

    # Configure performance logger verbosity
    configure_verbosity(args.performance_verbosity)
    logger.info(
        f"Performance logging verbosity set to: {args.performance_verbosity}"
    )

    # Enhanced progress bar defaults
    if not hasattr(args, "enhanced_progress_bar"):
        args.enhanced_progress_bar = True
    # Always show timing on every step by default (no alternation)
    if not hasattr(args, "alternate_perf_postfix"):
        args.alternate_perf_postfix = False

    # Parameter statistics logging
    if not hasattr(args, "log_param_stats"):
        args.log_param_stats = False  # Disabled by default
    if not hasattr(args, "param_stats_every_n_steps"):
        args.param_stats_every_n_steps = 100  # Log every 100 steps
    if not hasattr(args, "max_param_stats_logged"):
        args.max_param_stats_logged = 20  # Log top 20 parameters by norm

    # Per-source loss logging
    if not hasattr(args, "log_per_source_loss"):
        args.log_per_source_loss = False  # Disabled by default

    # Gradient norm logging
    if not hasattr(args, "log_gradient_norm"):
        args.log_gradient_norm = False  # Disabled by default

    # Extra train metrics (periodic)
    if not hasattr(args, "log_extra_train_metrics"):
        args.log_extra_train_metrics = True  # Enabled by default
    if not hasattr(args, "train_metrics_interval"):
        args.train_metrics_interval = 50  # Log every 50 steps by default

    # Report enabled features
    enabled_features = []

    if args.log_param_stats:
        enabled_features.append("Parameter Statistics")
        logger.info(f"Parameter statistics logging enabled:")
        logger.info(f"  - Logging every {args.param_stats_every_n_steps} steps")
        logger.info(
            f"  - Tracking top {args.max_param_stats_logged} parameters by norm"
        )
        logger.info(
            f"  - Will create TensorBoard metrics: param_norm/*, grad_norm/*, param_stats/*"
        )

    if args.log_per_source_loss:
        enabled_features.append("Per-Source Loss")
        logger.info("Per-source loss logging enabled:")
        logger.info("  - Will attempt to detect video vs image sources")
        logger.info("  - Will create TensorBoard metrics: loss/video, loss/image")

    if args.log_gradient_norm:
        enabled_features.append("Gradient Norm")
        logger.info("Gradient norm logging enabled:")
        logger.info("  - Will create TensorBoard metric: grad_norm")

    if enabled_features:
        logger.info(
            f"Advanced logging features enabled: {', '.join(enabled_features)}"
        )
    else:
        logger.info(
            "No advanced logging features enabled (use configure_advanced_logging to enable)"
        )