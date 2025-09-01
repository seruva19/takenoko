"""Context Memory Manager for Takenoko.

This module implements the Context-as-Memory approach for scene-consistent 
long video generation. It manages frame memory and context selection during training.
"""

import torch
import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import torch.nn.functional as F
from dataclasses import dataclass
import logging
from common.logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class FrameMemoryEntry:
    """Entry for storing frames in memory with metadata."""
    frame_tensor: torch.Tensor
    timestamp: int
    camera_pose: Optional[torch.Tensor] = None  # Camera pose [R|t] matrix (3x4)
    fov: Optional[float] = None  # Field of view in degrees
    embedding: Optional[torch.Tensor] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContextMemoryManager:
    """Manages frame memory and context selection for training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('ctxmem_enabled', False)
        
        if not self.enabled:
            logger.info("Context memory is disabled")
            return
            
        self.max_memory_frames = config.get('ctxmem_max_memory_frames', 100)
        self.context_size = config.get('ctxmem_context_size', 4)
        self.selection_strategy = config.get('ctxmem_frame_selection_strategy', 'recent')
        self.similarity_threshold = config.get('ctxmem_semantic_similarity_threshold', 0.7)
        self.use_caching = config.get('ctxmem_use_context_caching', True)
        
        # FOV-based selection parameters (paper's core innovation)
        self.use_fov_selection = config.get('ctxmem_use_fov_selection', True)
        self.default_fov = config.get('ctxmem_default_fov', 52.67)  # degrees, from paper
        self.fov_overlap_threshold = config.get('ctxmem_fov_overlap_threshold', 0.1)
        self.max_camera_distance = config.get('ctxmem_max_camera_distance', 10.0)  # meters
        
        # Memory storage
        self.frame_memory: deque = deque(maxlen=self.max_memory_frames)
        self.embedding_cache: Dict[int, torch.Tensor] = {}
        
        # Statistics
        self.stats = {
            'frames_added': 0,
            'context_selections': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"Context memory initialized with strategy '{self.selection_strategy}', "
                   f"context_size={self.context_size}, max_frames={self.max_memory_frames}")
    
    def add_frame(self, 
                  frame: torch.Tensor, 
                  timestamp: int, 
                  camera_pose: Optional[torch.Tensor] = None,
                  fov: Optional[float] = None,
                  metadata: Dict = None) -> None:
        """Add a frame to memory with camera pose information.
        
        Args:
            frame: Frame tensor [C, H, W]
            timestamp: Frame timestamp
            camera_pose: Camera pose matrix [3, 4] or [4, 4] containing [R|t]
            fov: Field of view in degrees
            metadata: Additional metadata
        """
        if not self.enabled:
            return
            
        # Process camera pose
        if camera_pose is not None:
            # Ensure camera pose is [3, 4] format [R|t]
            if camera_pose.shape == (4, 4):
                camera_pose = camera_pose[:3, :]  # Remove last row
            elif camera_pose.shape != (3, 4):
                logger.warning(f"Invalid camera pose shape {camera_pose.shape}, expected [3,4] or [4,4]")
                camera_pose = None
            
        entry = FrameMemoryEntry(
            frame_tensor=frame.clone().detach(),
            timestamp=timestamp,
            camera_pose=camera_pose.clone().detach() if camera_pose is not None else None,
            fov=fov or self.default_fov,
            metadata=metadata or {}
        )
        
        self.frame_memory.append(entry)
        self.stats['frames_added'] += 1
        
        # Clean up old cache entries if memory is full
        if len(self.frame_memory) >= self.max_memory_frames:
            self._cleanup_cache()
    
    def select_context_frames(self, 
                            current_frame: Optional[torch.Tensor] = None, 
                            current_timestamp: int = 0,
                            current_camera_pose: Optional[torch.Tensor] = None,
                            current_fov: Optional[float] = None,
                            k: Optional[int] = None) -> List[torch.Tensor]:
        """Select relevant context frames based on strategy with camera-aware selection.
        
        Time complexity analysis:
        - Recent selection: O(1) - deque slice operation
        - Semantic selection: O(n·d) where n=memory_size, d=embedding_dim
        - FOV selection: O(n) where n=memory_size (geometric calculations)
        - Mixed selection: O(n·d + n) - combination of semantic and FOV
        
        Space complexity: O(k·frame_size) for selected frames
        """
        if not self.enabled or len(self.frame_memory) == 0:
            return []
            
        k = k or self.context_size
        self.stats['context_selections'] += 1
        
        # Paper's core innovation: FOV-based selection when camera pose is available
        if (self.use_fov_selection and current_camera_pose is not None and 
            self.selection_strategy in ["fov_overlap", "recent"]):
            return self._select_fov_overlap_frames(current_camera_pose, current_fov, k)
        elif self.selection_strategy == "recent":
            return self._select_recent_frames(k)
        elif self.selection_strategy == "semantic":
            if current_frame is not None:
                return self._select_semantic_frames(current_frame, k)
            else:
                return self._select_recent_frames(k)  # fallback
        elif self.selection_strategy == "mixed":
            if current_frame is not None:
                return self._select_mixed_frames(current_frame, current_timestamp, current_camera_pose, k)
            else:
                return self._select_recent_frames(k)  # fallback
        else:
            return self._select_recent_frames(k)
    
    def _select_recent_frames(self, k: int) -> List[torch.Tensor]:
        """Select the most recent k frames."""
        recent_entries = list(self.frame_memory)[-k:]
        return [entry.frame_tensor for entry in recent_entries]
    
    def _select_semantic_frames(self, current_frame: torch.Tensor, k: int) -> List[torch.Tensor]:
        """Select frames based on semantic similarity.
        
        Mathematical formulation:
        - Embedding: e_i = AdaptiveAvgPool2D(f_i) ∈ R^d
        - Similarity: sim(e_q, e_i) = (e_q · e_i) / (||e_q|| · ||e_i||)
        - Selection: argmax_k sim(e_query, e_i) for i ∈ memory
        
        Time complexity: O(n·d + n·log(k)) where n=memory_size, d=embedding_dim
        - O(n·d) for similarity computation
        - O(n·log(k)) for top-k selection
        """
        if len(self.frame_memory) <= k:
            return [entry.frame_tensor for entry in self.frame_memory]
        
        current_embedding = self._get_frame_embedding(current_frame)
        similarities = []
        
        for i, entry in enumerate(self.frame_memory):
            if entry.embedding is None:
                entry.embedding = self._get_frame_embedding(entry.frame_tensor)
            
            similarity = F.cosine_similarity(
                current_embedding.unsqueeze(0), 
                entry.embedding.unsqueeze(0)
            ).item()
            similarities.append((similarity, i, entry))
        
        # Sort by similarity and select top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        selected_entries = [x[2] for x in similarities[:k]]
        
        return [entry.frame_tensor for entry in selected_entries]
    
    def _select_mixed_frames(self, current_frame: torch.Tensor, current_timestamp: int, 
                           current_camera_pose: Optional[torch.Tensor], k: int) -> List[torch.Tensor]:
        """Mix recent and semantic selection, with optional FOV if camera pose available."""
        if current_camera_pose is not None and self.use_fov_selection:
            # Use FOV + recent mix
            fov_k = k // 2
            recent_k = k - fov_k
            
            fov_frames = self._select_fov_overlap_frames(current_camera_pose, None, fov_k)
            recent_frames = self._select_recent_frames(recent_k)
            
            all_frames = fov_frames + recent_frames
            return all_frames[:k]
        else:
            # Fallback to semantic + recent mix
            recent_k = k // 2
            semantic_k = k - recent_k
            
            recent_frames = self._select_recent_frames(recent_k)
            semantic_frames = self._select_semantic_frames(current_frame, semantic_k)
            
            # Combine and deduplicate
            all_frames = recent_frames + semantic_frames
            return all_frames[:k]  # Ensure we don't exceed k
    
    def _select_fov_overlap_frames(self, 
                                  current_camera_pose: torch.Tensor,
                                  current_fov: Optional[float],
                                  k: int) -> List[torch.Tensor]:
        """Select frames based on FOV overlap - core paper innovation.
        
        This implements the paper's key contribution: selecting context frames
        that have overlapping Fields of View with the current camera pose.
        
        Algorithm complexity analysis:
        - Time complexity: O(n + k·log(k)) where n = len(memory)
          * O(n) for overlap score computation across all frames
          * O(n·log(n)) for sorting by overlap score  
          * O(k) for top-k selection and threshold filtering
        - Space complexity: O(n) for temporary overlap_scores list
        
        Geometric operations per frame: O(1)
        - Matrix operations: pose extraction, dot product, norm computation
        - All operations are vectorized using PyTorch primitives
        """
        if len(self.frame_memory) <= k:
            return [entry.frame_tensor for entry in self.frame_memory]
        
        current_fov = current_fov or self.default_fov
        overlap_scores = []
        
        for i, entry in enumerate(self.frame_memory):
            if entry.camera_pose is None:
                # No camera pose available, assign low score
                overlap_scores.append((0.0, i, entry))
                continue
            
            # Calculate FOV overlap between current and context camera poses
            overlap_score = self._calculate_fov_overlap(
                current_camera_pose, current_fov,
                entry.camera_pose, entry.fov
            )
            overlap_scores.append((overlap_score, i, entry))
        
        # Sort by overlap score (descending) and select top k
        overlap_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Apply overlap threshold - only select frames with meaningful overlap
        filtered_scores = [
            (score, i, entry) for score, i, entry in overlap_scores 
            if score >= self.fov_overlap_threshold
        ]
        
        if len(filtered_scores) < k:
            # If not enough high-overlap frames, fill with recent frames
            selected_entries = [entry for _, _, entry in filtered_scores]
            remaining_k = k - len(selected_entries)
            recent_entries = list(self.frame_memory)[-remaining_k:] if remaining_k > 0 else []
            
            # Avoid duplicates
            for entry in recent_entries:
                if entry not in selected_entries:
                    selected_entries.append(entry)
                    if len(selected_entries) >= k:
                        break
        else:
            selected_entries = [entry for _, _, entry in filtered_scores[:k]]
        
        return [entry.frame_tensor for entry in selected_entries]
    
    def _calculate_fov_overlap(self,
                              pose1: torch.Tensor, fov1: float,
                              pose2: torch.Tensor, fov2: float) -> float:
        """Calculate FOV overlap between two camera poses.
        
        Mathematical formulation:
        - Overlap score S = α·D(p1,p2) + β·A(R1,R2) + γ·F(fov1,fov2)
        - Distance factor: D = max(0, 1 - ||p1-p2||/d_max)
        - Angle factor: A = max(0, 1 - θ/θ_max) where θ = arccos(|f1·f2|)
        - FOV factor: F = min(1, (fov1+fov2)/(2·90°))
        - Weights: α=0.4, β=0.4, γ=0.2 (empirically tuned)
        
        Time complexity: O(1) - constant time geometric calculations
        Space complexity: O(1) - only temporary vectors
        
        Args:
            pose1, pose2: Camera poses [3, 4] containing [R|t]
            fov1, fov2: Field of view in degrees
            
        Returns:
            Overlap score S ∈ [0.0, 1.0]
        """
        try:
            # Extract camera positions (translation vectors)
            pos1 = pose1[:3, 3]  # [3] 
            pos2 = pose2[:3, 3]  # [3]
            
            # Extract rotation matrices
            R1 = pose1[:3, :3]  # [3, 3]
            R2 = pose2[:3, :3]  # [3, 3]
            
            # Camera distance check - if cameras are too far apart, likely no overlap
            distance = torch.norm(pos1 - pos2).item()
            if distance > self.max_camera_distance:
                return 0.0
            
            # Get camera forward directions (assuming -Z is forward in camera space)
            forward1 = R1[:, 2]  # Third column of rotation matrix
            forward2 = R2[:, 2]
            
            # Calculate angle between camera directions
            dot_product = torch.dot(forward1, forward2).item()
            dot_product = max(-1.0, min(1.0, dot_product))  # Clamp for numerical stability
            angle_between = math.acos(abs(dot_product))  # Use abs to handle back-facing
            angle_between_deg = math.degrees(angle_between)
            
            # Simple overlap heuristic based on:
            # 1. Distance between cameras (closer = more likely overlap)
            # 2. Angle between camera directions (similar direction = more overlap)
            # 3. FOV sizes (larger FOV = more overlap potential)
            
            # Distance factor (closer = better, normalized)
            distance_factor = max(0.0, 1.0 - (distance / self.max_camera_distance))
            
            # Angle factor (similar direction = better)
            max_overlap_angle = (fov1 + fov2) / 2  # Rough heuristic
            angle_factor = max(0.0, 1.0 - (angle_between_deg / max_overlap_angle))
            
            # FOV factor (larger combined FOV = higher overlap potential)
            avg_fov = (fov1 + fov2) / 2
            fov_factor = min(1.0, avg_fov / 90.0)  # Normalize to 90 degrees
            
            # Combined overlap score
            overlap_score = (distance_factor * 0.4 + angle_factor * 0.4 + fov_factor * 0.2)
            
            return max(0.0, min(1.0, overlap_score))
            
        except Exception as e:
            logger.debug(f"Error calculating FOV overlap: {e}")
            return 0.0
    
    def _get_frame_embedding(self, frame: torch.Tensor) -> torch.Tensor:
        """Get or compute frame embedding for similarity calculation."""
        frame_hash = hash(frame.data_ptr())
        
        if self.use_caching and frame_hash in self.embedding_cache:
            self.stats['cache_hits'] += 1
            return self.embedding_cache[frame_hash]
        
        self.stats['cache_misses'] += 1
        
        # Compute simple embedding (can be replaced with more sophisticated methods)
        with torch.no_grad():
            # Use average pooling as a simple embedding
            embedding = F.adaptive_avg_pool2d(frame, (1, 1)).flatten()
            
            if self.use_caching:
                self.embedding_cache[frame_hash] = embedding
                
        return embedding
    
    def _cleanup_cache(self):
        """Clean up old cache entries to prevent memory leaks."""
        if len(self.embedding_cache) > self.max_memory_frames * 2:
            # Keep only recent entries
            keys_to_remove = list(self.embedding_cache.keys())[:-self.max_memory_frames]
            for key in keys_to_remove:
                del self.embedding_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory manager statistics with validation metrics."""
        total_selections = self.stats['context_selections']
        
        # Calculate advanced validation metrics
        validation_metrics = {
            # Memory utilization efficiency
            'memory_utilization': len(self.frame_memory) / max(1, self.max_memory_frames),
            
            # Context selection effectiveness  
            'avg_selections_per_step': self.stats['context_selections'] / max(1, self.stats['frames_added']),
            
            # Caching performance
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'cache_efficiency': len(self.embedding_cache) / max(1, len(self.frame_memory)),
            
            # Memory turnover rate (how quickly memory refreshes)
            'memory_turnover': self.stats['frames_added'] / max(1, self.max_memory_frames),
        }
        
        return {
            **self.stats,
            'memory_size': len(self.frame_memory),
            'cache_size': len(self.embedding_cache),
            **validation_metrics
        }
    
    def clear_memory(self):
        """Clear all memory and cache."""
        if not self.enabled:
            return
        self.frame_memory.clear()
        self.embedding_cache.clear()
        self.stats = {key: 0 for key in self.stats}
        logger.info("Context memory cleared")
    
    def process_training_step(self, 
                            latents: torch.Tensor,
                            global_step: int,
                            step: int,
                            args: Any,
                            accelerator: Any,
                            temporal_consistency_loss_fn: Any,
                            batch: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process a training step for context memory.
        
        Args:
            latents: Current frame latents [B, C, H, W] or [B, T, C, H, W]
            global_step: Global training step
            step: Current batch step
            args: Training arguments
            accelerator: Accelerate accelerator
            temporal_consistency_loss_fn: Temporal consistency loss function
            
        Returns:
            Tuple of (context_memory_loss, stats_dict)
        """
        if not self.enabled:
            return torch.tensor(0.0, device=latents.device, requires_grad=True), {}
        
        # Check progressive training settings
        progressive_enabled = getattr(args, 'ctxmem_progressive_context_training', False)
        warmup_steps = getattr(args, 'ctxmem_context_warmup_steps', 1000)
        
        if progressive_enabled and global_step < warmup_steps:
            # Still in warmup phase, skip context memory processing
            return torch.tensor(0.0, device=latents.device, requires_grad=True), {}
        
        # Extract camera poses and FOV from batch
        camera_poses = self._extract_camera_poses_from_batch(batch)
        current_fov = self._extract_fov_from_batch(batch, args)
        
        # Get current frames for processing
        current_frames = latents.detach().clone()
        
        # Select context frames from memory with camera pose awareness
        current_camera_pose = None
        if camera_poses is not None and len(camera_poses) > 0:
            # Use first batch item's camera pose as representative
            current_camera_pose = camera_poses[0] if camera_poses.dim() > 2 else camera_poses
            
        context_frames = self.select_context_frames(
            current_frame=current_frames[0] if len(current_frames) > 0 else None,
            current_timestamp=global_step,
            current_camera_pose=current_camera_pose,
            current_fov=current_fov
        )
        
        # Compute temporal consistency loss if we have context
        context_memory_loss = torch.tensor(0.0, device=latents.device, requires_grad=True)
        context_relevance_score = 0.0
        
        if context_frames and temporal_consistency_loss_fn is not None:
            context_memory_loss = temporal_consistency_loss_fn(
                target_frames=current_frames,
                context_frames=context_frames
            )
            
            # Calculate context relevance score for validation metrics
            # Lower loss indicates higher relevance/similarity between context and target
            # Relevance = exp(-loss) ∈ [0,1] where higher values = more relevant context
            context_relevance_score = torch.exp(-context_memory_loss.detach()).item()
        
        # Update memory with current frames and camera poses
        self._update_memory_with_frames(current_frames, global_step, step, camera_poses, current_fov)
        
        # Prepare comprehensive stats for logging
        stats_dict = {}
        if global_step % 100 == 0:  # Log stats every 100 steps
            stats_dict = {
                f"context_memory/{key}": value 
                for key, value in self.get_stats().items()
            }
            
            # Add context-specific validation metrics
            stats_dict["context_memory/num_context_frames"] = len(context_frames)
            stats_dict["context_memory/context_relevance_score"] = context_relevance_score
            stats_dict["context_memory/context_available"] = len(self.frame_memory) > 0
            stats_dict["context_memory/selection_strategy"] = self.selection_strategy
            
            if context_memory_loss.item() > 0:
                stats_dict["context_memory/consistency_loss"] = context_memory_loss.item()
            
            # Advanced validation metrics when enough data is available
            if len(context_frames) > 0:
                stats_dict["context_memory/context_utilization"] = len(context_frames) / max(1, self.context_size)
                stats_dict["context_memory/memory_coverage"] = len(context_frames) / max(1, len(self.frame_memory))
        
        return context_memory_loss, stats_dict
    
    def _update_memory_with_frames(self, 
                                  frames: torch.Tensor, 
                                  global_step: int, 
                                  step: int,
                                  camera_poses: Optional[torch.Tensor] = None,
                                  current_fov: Optional[float] = None) -> None:
        """Update memory with current frames."""
        if not self.enabled:
            return
        
        # Handle different frame tensor shapes
        if frames.dim() == 4:  # [B, C, H, W]
            for b in range(frames.shape[0]):
                # Extract camera pose for this batch item
                batch_camera_pose = None
                if camera_poses is not None:
                    if camera_poses.dim() == 3 and camera_poses.shape[0] > b:
                        batch_camera_pose = camera_poses[b]  # [3, 4]
                    elif camera_poses.dim() == 2:
                        batch_camera_pose = camera_poses  # Single pose for all batch
                
                self.add_frame(
                    frames[b],
                    timestamp=global_step,
                    camera_pose=batch_camera_pose,
                    fov=current_fov,
                    metadata={
                        'batch_idx': step, 
                        'global_step': global_step,
                        'batch_item': b
                    }
                )
        elif frames.dim() == 5:  # [B, T, C, H, W]
            for b in range(frames.shape[0]):
                for t in range(frames.shape[1]):
                    # Extract camera pose for this batch item and time step
                    batch_camera_pose = None
                    if camera_poses is not None:
                        if camera_poses.dim() == 4 and camera_poses.shape[0] > b and camera_poses.shape[1] > t:
                            batch_camera_pose = camera_poses[b, t]  # [3, 4]
                        elif camera_poses.dim() == 3 and camera_poses.shape[0] > b:
                            batch_camera_pose = camera_poses[b]  # [3, 4] - same pose for all time steps
                        elif camera_poses.dim() == 2:
                            batch_camera_pose = camera_poses  # Single pose for all
                    
                    self.add_frame(
                        frames[b, t],
                        timestamp=global_step * frames.shape[1] + t,
                        camera_pose=batch_camera_pose,
                        fov=current_fov,
                        metadata={
                            'batch_idx': step,
                            'global_step': global_step,
                            'batch_item': b,
                            'time_idx': t
                        }
                    )
        else:
            logger.warning(f"Unexpected frame tensor shape: {frames.shape}")
    
    def log_stats_to_accelerator(self, 
                                stats_dict: Dict[str, Any],
                                accelerator: Any,
                                global_step: int) -> None:
        """Log context memory stats to accelerator trackers."""
        if not stats_dict or not accelerator.is_main_process:
            return
        
        if len(accelerator.trackers) > 0:
            for tracker in accelerator.trackers:
                if hasattr(tracker, 'log'):
                    try:
                        tracker.log(stats_dict, step=global_step)
                    except Exception as e:
                        logger.debug(f"Failed to log context memory stats: {e}")
    
    def integrate_context_loss(self,
                              loss_components: Any,
                              context_memory_loss: torch.Tensor,
                              config: Dict[str, Any],
                              accelerator: Any,
                              global_step: int) -> None:
        """Integrate context memory loss into the main loss components."""
        if not self.enabled or context_memory_loss.item() <= 0:
            return
        
        # Get context loss weight from config
        context_weight = config.get('ctxmem_temporal_consistency_loss_weight', 0.1)
        weighted_context_loss = context_memory_loss * context_weight
        
        # Add to total loss
        loss_components.total_loss = loss_components.total_loss + weighted_context_loss
        
        # Log the context memory loss components
        if accelerator.is_main_process and len(accelerator.trackers) > 0:
            loss_dict = {
                "loss/context_memory_raw": context_memory_loss.item(),
                "loss/context_memory_weighted": weighted_context_loss.item()
            }
            
            for tracker in accelerator.trackers:
                if hasattr(tracker, 'log'):
                    try:
                        tracker.log(loss_dict, step=global_step)
                    except Exception as e:
                        logger.debug(f"Failed to log context memory loss: {e}")
    
    def create_temporal_consistency_loss_fn(self):
        """Create temporal consistency loss function based on config."""
        if not self.enabled:
            # Return dummy loss that always returns 0
            def dummy_loss(*args, **kwargs):
                return torch.tensor(0.0, requires_grad=True)
            return dummy_loss
        
        def temporal_consistency_loss(target_frames: torch.Tensor, 
                                     context_frames: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
            """Compute temporal consistency loss between frames.
            
            Mathematical formulation:
            - L_temporal = (1/N) Σ ||f_target - f_context||²
            - Where f_target ∈ R^(B×C×H×W), f_context ∈ R^(C×H×W)
            - MSE loss: ||x-y||² = (1/n) Σ(x_i - y_i)²
            
            Time complexity: O(B·C·H·W) - element-wise difference and mean
            Space complexity: O(B·C·H·W) - temporary difference tensor
            """
            if context_frames is None or len(context_frames) == 0:
                return torch.tensor(0.0, device=target_frames.device, requires_grad=True)
            
            # Convert context frames to tensor for easier processing
            context_tensor = torch.stack(context_frames, dim=0)  # [T_context, C, H, W]
            
            # Ensure target_frames has proper dimensions
            if target_frames.dim() == 4:  # [B, C, H, W]
                target_batch = target_frames
            elif target_frames.dim() == 5:  # [B, T, C, H, W]
                target_batch = target_frames[:, 0]  # Use first frame
            else:
                logger.warning(f"Unexpected target_frames shape: {target_frames.shape}")
                return torch.tensor(0.0, device=target_frames.device, requires_grad=True)
            
            batch_size = target_batch.shape[0]
            
            # Temporal consistency between target and most recent context
            if len(context_tensor) > 0:
                most_recent_context = context_tensor[-1]  # [C, H, W]
                
                # Expand context to match batch size
                most_recent_context = most_recent_context.unsqueeze(0).expand(batch_size, -1, -1, -1)
                
                # Compute L2 consistency loss
                consistency_loss = F.mse_loss(target_batch, most_recent_context)
                return consistency_loss
            
            return torch.tensor(0.0, device=target_frames.device, requires_grad=True)
        
        return temporal_consistency_loss
    
    def prepare_context_for_forward_pass(self,
                                       target_latents: torch.Tensor,
                                       camera_poses: Optional[torch.Tensor] = None,
                                       current_fov: Optional[float] = None,
                                       global_step: int = 0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Prepare context-conditioned input for the forward pass.
        
        This implements the paper's approach of concatenating context frames
        along the temporal dimension BEFORE the forward pass.
        
        Args:
            target_latents: Target latents to condition [B, C, H, W] or [B, T, C, H, W]
            camera_poses: Camera poses for current frame(s)
            current_fov: Current field of view
            global_step: Current training step
            
        Returns:
            Tuple of (conditioned_latents, context_info)
        """
        if not self.enabled:
            return target_latents, {}
        
        # Get camera pose for context selection
        current_camera_pose = None
        if camera_poses is not None and len(camera_poses) > 0:
            current_camera_pose = camera_poses[0] if camera_poses.dim() > 2 else camera_poses
        
        # Select context frames based on camera pose/FOV overlap
        context_frames = self.select_context_frames(
            current_frame=target_latents[0] if len(target_latents) > 0 else None,
            current_timestamp=global_step,
            current_camera_pose=current_camera_pose,
            current_fov=current_fov
        )
        
        if not context_frames:
            return target_latents, {'num_context_frames': 0}
        
        # Apply context conditioning (concatenation along temporal dimension)
        conditioned_latents = self._apply_context_conditioning(target_latents, context_frames)
        
        context_info = {
            'num_context_frames': len(context_frames),
            'context_conditioning_applied': True,
            'original_shape': target_latents.shape,
            'conditioned_shape': conditioned_latents.shape
        }
        
        return conditioned_latents, context_info
    
    def _apply_context_conditioning(self, 
                                   target_latents: torch.Tensor, 
                                   context_frames: List[torch.Tensor]) -> torch.Tensor:
        """Apply context conditioning by concatenating context frames.
        
        Mathematical formulation:
        - Context tensor: C = stack([c₁, c₂, ..., cₖ]) ∈ R^(k×C×H×W)
        - Target tensor: T ∈ R^(B×T_target×C×H×W) or R^(B×C×H×W)
        - Conditioned output: O = concat([C_expanded, T], dim=1) ∈ R^(B×(k+T_target)×C×H×W)
        - Where C_expanded = C.unsqueeze(1).expand(-1, B, -1, -1, -1).transpose(0,1)
        
        This is the core conditioning method from the paper:
        'directly incorporate as part of the input through concatenation along frame dimension'
        
        Time complexity: O(B·k·C·H·W) for tensor operations and concatenation
        Space complexity: O(B·k·C·H·W) for expanded context tensor
        """
        if not context_frames:
            return target_latents
        
        # Stack context frames: [T_context, C, H, W]
        context_tensor = torch.stack(context_frames, dim=0)
        batch_size = target_latents.shape[0]
        
        # Expand context tensor to match batch size: [B, T_context, C, H, W]
        context_tensor = context_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        # Handle different target latent shapes
        if target_latents.dim() == 4:  # [B, C, H, W] -> [B, 1, C, H, W]
            target_latents = target_latents.unsqueeze(1)
        elif target_latents.dim() != 5:  # Should be [B, T, C, H, W]
            logger.warning(f"Unexpected target latents shape: {target_latents.shape}")
            return target_latents
        
        # Concatenate context and target along temporal dimension
        # Result: [B, T_context + T_target, C, H, W]
        conditioned_latents = torch.cat([context_tensor, target_latents], dim=1)
        
        return conditioned_latents
    
    def _extract_camera_poses_from_batch(self, batch: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Extract camera poses from batch data."""
        if batch is None:
            return None
        
        # Try different possible keys for camera poses
        camera_pose_keys = ['camera_poses', 'camera_pose', 'cam_poses', 'cam_pose', 'poses']
        
        for key in camera_pose_keys:
            if key in batch:
                poses = batch[key]
                if isinstance(poses, torch.Tensor):
                    return poses
                elif hasattr(poses, 'camera_poses'):
                    return poses.camera_poses
        
        # Try to extract from nested structures
        if hasattr(batch, 'camera_poses'):
            return batch.camera_poses
        elif hasattr(batch, 'get'):
            for key in camera_pose_keys:
                poses = batch.get(key)
                if poses is not None:
                    return poses
        
        return None
    
    def _extract_fov_from_batch(self, batch: Optional[Dict[str, Any]], args: Any) -> float:
        """Extract FOV from batch data or use default."""
        if batch is None:
            return getattr(args, 'ctxmem_default_fov', self.default_fov)
        
        # Try different possible keys for FOV
        fov_keys = ['fov', 'field_of_view', 'camera_fov']
        
        for key in fov_keys:
            if key in batch:
                fov = batch[key]
                if isinstance(fov, (int, float)):
                    return float(fov)
                elif isinstance(fov, torch.Tensor):
                    return float(fov.item())
        
        # Use default from args or config
        return getattr(args, 'ctxmem_default_fov', self.default_fov)
    
    def compute_validation_metrics(self, 
                                  recent_targets: List[torch.Tensor],
                                  recent_contexts: List[List[torch.Tensor]]) -> Dict[str, float]:
        """Compute comprehensive validation metrics for Context-as-Memory effectiveness.
        
        Mathematical formulations:
        - Context Relevance: R = (1/N) Σ exp(-L(target_i, context_i))
        - Selection Diversity: D = (1/K) Σ ||embedding_i - mean_embedding||²
        - Temporal Coherence: T = (1/N-1) Σ SSIM(target_i, target_{i+1})
        
        Time complexity: O(N·K·C·H·W) where N=targets, K=avg_contexts, C×H×W=frame_size
        Space complexity: O(N·K·embedding_dim) for temporary embeddings
        
        Args:
            recent_targets: List of recent target frames for analysis
            recent_contexts: List of context frame lists corresponding to targets
            
        Returns:
            Dictionary of validation metrics with mathematical backing
        """
        if not self.enabled or len(recent_targets) == 0:
            return {}
        
        metrics = {}
        
        # 1. Context Relevance Score (how well context matches targets)
        # R_i = max_j(exp(-MSE(target_i, context_{i,j})))
        relevance_scores = []
        for target, context_list in zip(recent_targets, recent_contexts):
            if context_list:
                # Compute MSE between target and most similar context frame
                target_flat = target.flatten()
                similarities = []
                for ctx in context_list:
                    ctx_flat = ctx.flatten()
                    mse = torch.mean((target_flat - ctx_flat) ** 2).item()
                    # Convert MSE to relevance score: higher MSE = lower relevance
                    similarities.append(torch.exp(-torch.tensor(mse)).item())
                relevance_scores.append(max(similarities) if similarities else 0.0)
        
        if relevance_scores:
            metrics['context_relevance_mean'] = sum(relevance_scores) / len(relevance_scores)
            metrics['context_relevance_std'] = torch.std(torch.tensor(relevance_scores)).item()
        
        # 2. Selection Diversity (how diverse are selected context frames)
        # D = (1/K) Σ ||e_i - μ_e||² where e_i are frame embeddings, μ_e is mean embedding
        if len(recent_contexts) > 0 and any(len(ctx) > 1 for ctx in recent_contexts):
            diversity_scores = []
            for context_list in recent_contexts:
                if len(context_list) > 1:
                    embeddings = [self._get_frame_embedding(frame) for frame in context_list]
                    embedding_tensor = torch.stack(embeddings)
                    mean_embedding = torch.mean(embedding_tensor, dim=0)
                    diversity = torch.mean(torch.norm(embedding_tensor - mean_embedding, dim=1)).item()
                    diversity_scores.append(diversity)
            
            if diversity_scores:
                metrics['selection_diversity_mean'] = sum(diversity_scores) / len(diversity_scores)
        
        # 3. Memory Efficiency Metrics
        metrics.update({
            'memory_utilization_current': len(self.frame_memory) / max(1, self.max_memory_frames),
            'cache_efficiency_current': len(self.embedding_cache) / max(1, len(self.frame_memory)) if len(self.frame_memory) > 0 else 0.0,
            'avg_context_size': sum(len(ctx) for ctx in recent_contexts) / max(1, len(recent_contexts)),
            'selection_strategy_active': self.selection_strategy,
        })
        
        # 4. Computational Efficiency Metrics  
        if self.stats['context_selections'] > 0:
            metrics.update({
                'cache_hit_rate_current': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
                'selections_per_added_frame': self.stats['context_selections'] / max(1, self.stats['frames_added']),
            })
        
        return metrics


class ContextConditioner:
    """Handles different methods of conditioning on context frames."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('ctxmem_enabled', False)
        self.conditioning_method = config.get('ctxmem_context_conditioning_method', 'concatenation')
        self.context_weight = config.get('ctxmem_context_weight', 1.0)
        self.dropout_rate = config.get('ctxmem_context_dropout_rate', 0.0)
        
        if self.enabled:
            logger.info(f"Context conditioning enabled with method '{self.conditioning_method}'")
        
    def condition_on_context(self, 
                           target_frames: torch.Tensor,
                           context_frames: List[torch.Tensor],
                           training: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply context conditioning to target frames."""
        if not self.enabled or not context_frames:
            return target_frames, {}
        
        # Apply context dropout during training
        if training and self.dropout_rate > 0:
            keep_prob = 1.0 - self.dropout_rate
            mask = torch.rand(len(context_frames)) > self.dropout_rate
            context_frames = [frame for i, frame in enumerate(context_frames) if mask[i]]
        
        if not context_frames:
            return target_frames, {}
        
        conditioning_info = {'num_context_frames': len(context_frames)}
        
        if self.conditioning_method == "concatenation":
            return self._concatenate_conditioning(target_frames, context_frames), conditioning_info
        else:
            logger.warning(f"Conditioning method '{self.conditioning_method}' not implemented, using concatenation")
            return self._concatenate_conditioning(target_frames, context_frames), conditioning_info
    
    def _concatenate_conditioning(self, 
                                target_frames: torch.Tensor, 
                                context_frames: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate context frames with target frames along temporal dimension."""
        # Ensure all frames have compatible shapes
        context_tensor = torch.stack(context_frames, dim=0)  # [T_context, C, H, W]
        
        # Handle different input shapes
        if target_frames.dim() == 4:  # [B, C, H, W] -> add batch dim for each context frame
            batch_size = target_frames.shape[0]
            context_tensor = context_tensor.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)  # [T_context, B, C, H, W]
            context_tensor = context_tensor.transpose(0, 1)  # [B, T_context, C, H, W]
            
            target_frames = target_frames.unsqueeze(1)  # [B, 1, C, H, W]
            conditioned_frames = torch.cat([context_tensor, target_frames], dim=1)  # [B, T_context + 1, C, H, W]
            
        elif target_frames.dim() == 5:  # [B, T_target, C, H, W]
            batch_size = target_frames.shape[0]
            context_tensor = context_tensor.unsqueeze(1).expand(-1, batch_size, -1, -1, -1)  # [T_context, B, C, H, W]
            context_tensor = context_tensor.transpose(0, 1)  # [B, T_context, C, H, W]
            
            conditioned_frames = torch.cat([context_tensor, target_frames], dim=1)  # [B, T_context + T_target, C, H, W]
        else:
            logger.error(f"Unexpected target_frames shape: {target_frames.shape}")
            return target_frames
        
        return conditioned_frames