"""Weight Dynamics Analysis Utilities

This module provides comprehensive analysis tools for monitoring weight evolution
during neural network training. Supports research into training dynamics,
parameter plasticity, and learning progression across different model components.
"""

import torch
from typing import Dict, Optional, Tuple
import logging

from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class WeightDynamicsAnalyzer:
    """Advanced analysis tool for monitoring weight evolution and training dynamics.

    This class provides comprehensive statistical analysis of parameter changes
    during training, enabling research into:
    - Training efficiency across different layers
    - Parameter plasticity and adaptation rates
    - Learning progression patterns
    - Model component specialization
    """

    def __init__(self):
        self.initial_weights = {}
        self.layer_weight_counts = {}
        self.analysis_enabled = False

    def initialize_baseline_statistics(self, model: torch.nn.Module) -> None:
        """Initialize baseline weight statistics for comparative analysis."""
        self.initial_weights = {}
        self.layer_weight_counts = {}

        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store initial weight statistics
                self.initial_weights[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'abs_mean': param.data.abs().mean().item(),
                    'shape': param.shape
                }
                trainable_params += param.numel()

                # Count parameters by layer type
                layer_type = name.split('.')[0] if '.' in name else name
                if layer_type not in self.layer_weight_counts:
                    self.layer_weight_counts[layer_type] = 0
                self.layer_weight_counts[layer_type] += param.numel()

            total_params += param.numel()

        self.analysis_enabled = True

        logger.info(f"📊 Weight Dynamics Analysis Initialized:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Training coverage: {100 * trainable_params / total_params:.1f}%")
        logger.info(f"   Component types monitored: {len(self.layer_weight_counts)}")

    def analyze_parameter_evolution(self, model: torch.nn.Module, step: int) -> dict:
        """Analyze parameter evolution patterns and plasticity across model components."""
        if not self.analysis_enabled or not self.initial_weights:
            return {}

        changes = {}
        layer_changes = {}
        unchanged_params = []

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.initial_weights:
                initial = self.initial_weights[name]
                current_mean = param.data.mean().item()
                current_std = param.data.std().item()
                current_abs_mean = param.data.abs().mean().item()

                # Calculate relative changes
                mean_change = abs(current_mean - initial['mean'])
                std_change = abs(current_std - initial['std'])
                abs_mean_change = abs(current_abs_mean - initial['abs_mean'])

                # Normalize by initial values to get relative change
                rel_mean_change = mean_change / (abs(initial['mean']) + 1e-8)
                rel_std_change = std_change / (initial['std'] + 1e-8)
                rel_abs_change = abs_mean_change / (initial['abs_mean'] + 1e-8)

                changes[name] = {
                    'mean_change': mean_change,
                    'rel_mean_change': rel_mean_change,
                    'rel_std_change': rel_std_change,
                    'rel_abs_change': rel_abs_change,
                    'max_rel_change': max(rel_mean_change, rel_std_change, rel_abs_change)
                }

                # Track by layer type
                layer_type = name.split('.')[0] if '.' in name else name
                if layer_type not in layer_changes:
                    layer_changes[layer_type] = []
                layer_changes[layer_type].append(changes[name]['max_rel_change'])

                # Flag parameters with minimal adaptation (conservative threshold for research)
                if changes[name]['max_rel_change'] < 1e-7:
                    unchanged_params.append(name)

        # Summarize by layer type
        layer_summary = {}
        for layer_type, change_list in layer_changes.items():
            layer_summary[layer_type] = {
                'avg_change': sum(change_list) / len(change_list),
                'max_change': max(change_list),
                'min_change': min(change_list),
                'param_count': len(change_list)
            }

        return {
            'step': step,
            'total_params_analyzed': len(changes),
            'minimal_adaptation_params': unchanged_params,
            'minimal_adaptation_count': len(unchanged_params),
            'component_plasticity_summary': layer_summary
        }

    def log_dynamics_analysis_summary(self, analysis_result: dict) -> None:
        """Log comprehensive summary of parameter evolution analysis."""
        if not analysis_result:
            return

        step = analysis_result['step']
        total_analyzed = analysis_result['total_params_analyzed']
        minimal_count = analysis_result['minimal_adaptation_count']
        plasticity_summary = analysis_result['component_plasticity_summary']

        logger.info(f"🔬 Parameter Evolution Analysis (Step {step}):")
        logger.info(f"   Parameters analyzed: {total_analyzed:,}")
        logger.info(f"   Minimal adaptation: {minimal_count} ({100 * minimal_count / total_analyzed:.1f}%)")

        if minimal_count > 0:
            logger.info(f"📊 {minimal_count} parameters showing minimal adaptation (within research threshold)")

        # Log component plasticity rankings
        if plasticity_summary:
            sorted_components = sorted(
                plasticity_summary.items(),
                key=lambda x: x[1]['avg_change'],
                reverse=True
            )

            logger.info("   Component plasticity ranking:")
            for component_type, stats in sorted_components[:5]:
                logger.info(
                    f"     {component_type}: avg={stats['avg_change']:.2e}, "
                    f"max={stats['max_change']:.2e}, params={stats['param_count']}"
                )

    def analyze_gradient_dynamics(self, model: torch.nn.Module, step: int) -> dict:
        """Analyze gradient flow patterns and magnitudes across model architecture."""
        grad_stats = {}
        no_grad_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    grad_mean = param.grad.data.mean().item()
                    grad_std = param.grad.data.std().item()

                    grad_stats[name] = {
                        'grad_norm': grad_norm,
                        'grad_mean': grad_mean,
                        'grad_std': grad_std,
                        'has_grad': True
                    }
                else:
                    grad_stats[name] = {'has_grad': False}
                    no_grad_params.append(name)

        return {
            'step': step,
            'total_params': len(grad_stats),
            'no_grad_params': no_grad_params,
            'no_grad_count': len(no_grad_params),
            'grad_stats': grad_stats
        }

    def log_gradient_flow_summary(self, grad_result: dict) -> None:
        """Log a summary of gradient flow verification."""
        if not grad_result:
            return

        step = grad_result['step']
        total_params = grad_result['total_params']
        no_grad_count = grad_result['no_grad_count']
        no_grad_params = grad_result['no_grad_params']

        logger.info(f"🌊 Gradient Flow Verification (Step {step}):")
        logger.info(f"   Parameters with gradients: {total_params - no_grad_count:,}/{total_params:,}")

        if no_grad_count > 0:
            logger.warning(f"⚠️ Found {no_grad_count} parameters without gradients:")
            for param_name in no_grad_params[:10]:  # Show first 10
                logger.warning(f"     - {param_name}")
            if no_grad_count > 10:
                logger.warning(f"     ... and {no_grad_count - 10} more")


def quick_weight_check(model: torch.nn.Module, step: int) -> Tuple[int, int, float]:
    """Quick check to count trainable parameters and estimate if they're changing.

    Returns:
        Tuple of (total_params, trainable_params, avg_weight_magnitude)
    """
    total_params = 0
    trainable_params = 0
    weight_sum = 0.0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            weight_sum += param.data.abs().sum().item()

    avg_weight_magnitude = weight_sum / trainable_params if trainable_params > 0 else 0.0

    return total_params, trainable_params, avg_weight_magnitude


def log_parameter_summary(model: torch.nn.Module) -> None:
    """Log a quick summary of model parameters for verification."""
    total, trainable, avg_mag = quick_weight_check(model, 0)

    logger.info(f"📈 Parameter Summary:")
    logger.info(f"   Total parameters: {total:,}")
    logger.info(f"   Trainable parameters: {trainable:,}")
    logger.info(f"   Trainable percentage: {100 * trainable / total:.1f}%")
    logger.info(f"   Average weight magnitude: {avg_mag:.6f}")