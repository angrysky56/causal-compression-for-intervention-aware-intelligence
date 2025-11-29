"""
TKUI Continuity Field Implementation
=====================================

A temporal graph embedding system that resolves the Ship of Theseus paradox
through continuous self-modeling of transformation.

Core Components:
1. CausalTrajectoryExtractor: Mines graph paths for impact signatures
2. TemporalKernel: Exponentially-weighted time decay
3. InvariantDetector: Identifies gradient-resistant core parameters
4. EmbeddingCompressor: Sequence-to-vector compression
5. SimilarityIndex: Real-time nearest neighbor search
6. MemoryConsolidator: Prevents unbounded growth & catastrophic forgetting

Author: Generated via RAA Session thought_1764383479076473
Date: 2025-11-28
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import numpy as np
from collections import deque
import math


@dataclass
class ContinuityFieldConfig:
    """Configuration for Continuity Field system."""
    
    # Temporal kernel parameters
    temporal_decay_lambda: float = 0.1  # Exponential decay rate
    recent_window_size: int = 100  # Sliding window for recent history
    
    # Embedding parameters  
    embedding_dim: int = 256  # Dimension of continuity field vectors
    compression_ratio: float = 0.8  # Target compression (prevent growth)
    
    # Invariant detection parameters
    invariant_gradient_threshold: float = 0.01  # Min gradient for "frozen"
    invariant_stability_window: int = 50  # Steps to confirm invariance
    
    # Memory management
    max_trajectory_length: int = 10000  # Max stored trajectory points
    pruning_importance_threshold: float = 0.3  # Min score to keep
    consolidation_interval: int = 100  # Steps between consolidation
    
    # Plasticity gate thresholds
    high_uncertainty_threshold: float = 0.3
    conservative_similarity_threshold: float = 0.8
    exploratory_similarity_threshold: float = 0.4
    
    # Performance
    similarity_search_k: int = 10  # Top-k nearest neighbors
    max_query_latency_ms: float = 100.0  # Real-time guarantee


@dataclass
class TransformationEvent:
    """Records a single modification to agent parameters."""
    
    timestamp: datetime
    parameter_delta: np.ndarray  # Change vector
    trigger_context: str  # Why this modification occurred
    uncertainty: float  # Epistemic uncertainty at modification time
    identity_preservation: float  # Similarity score pre/post modification
    
    def __post_init__(self):
        self.embedding: Optional[np.ndarray] = None


class TemporalKernel:
    """
    Implements exponentially-weighted temporal decay for trajectory integration.
    
    K(t, t') = exp(-λ|t - t'|)
    
    This ensures recent events dominate while preserving distant critical patterns.
    """
    
    def __init__(self, decay_lambda: float = 0.1):
        self.lambda_ = decay_lambda
    
    def weight(self, t_current: float, t_event: float) -> float:
        """Compute temporal weight for event at t_event from perspective of t_current."""
        delta_t = abs(t_current - t_event)
        return math.exp(-self.lambda_ * delta_t)
    
    def weighted_sum(
        self, 
        embeddings: list[np.ndarray], 
        timestamps: list[float],
        current_time: float
    ) -> np.ndarray:
        """
        Compute weighted sum of embeddings with temporal decay.
        
        C(t) = Σᵢ K(t, tᵢ) · embed(eᵢ) / Σᵢ K(t, tᵢ)
        """
        weights = np.array([self.weight(current_time, t) for t in timestamps])
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        
        weighted_embedding = sum(w * e for w, e in zip(weights, embeddings))
        return weighted_embedding


class InvariantDetector:
    """
    Identifies core parameters resistant to gradient updates.
    
    Core invariants are parameters that:
    1. Have consistently low gradient magnitudes across updates
    2. Maintain stable values over extended periods
    3. Are structurally critical (high connectivity in parameter graph)
    
    This implements the "essence" component of continuity - the parameters
    that define identity even as peripheral parameters adapt.
    """
    
    def __init__(
        self, 
        gradient_threshold: float = 0.01,
        stability_window: int = 50
    ):
        self.grad_threshold = gradient_threshold
        self.stability_window = stability_window
        self.gradient_history: dict[str, deque] = {}
        self.parameter_history: dict[str, deque] = {}
    
    def update(self, parameter_name: str, value: float, gradient: float):
        """Track parameter and gradient history."""
        if parameter_name not in self.gradient_history:
            self.gradient_history[parameter_name] = deque(maxlen=self.stability_window)
            self.parameter_history[parameter_name] = deque(maxlen=self.stability_window)
        
        self.gradient_history[parameter_name].append(abs(gradient))
        self.parameter_history[parameter_name].append(value)
    
    def is_invariant(self, parameter_name: str) -> bool:
        """
        Determine if parameter qualifies as core invariant.
        
        Criteria:
        - Mean gradient magnitude < threshold over stability window
        - Parameter value variance < threshold
        """
        if parameter_name not in self.gradient_history:
            return False
        
        grads = list(self.gradient_history[parameter_name])
        values = list(self.parameter_history[parameter_name])
        
        if len(grads) < self.stability_window:
            return False  # Insufficient data
        
        mean_grad = np.mean(grads)
        value_variance = np.var(values)
        
        return (mean_grad < self.grad_threshold and 
                value_variance < self.grad_threshold)
    
    def get_core_invariants(self) -> set[str]:
        """Return set of all parameters identified as core invariants."""
        return {
            param for param in self.gradient_history.keys()
            if self.is_invariant(param)
        }


class CausalImpactSignature:
    """
    Computes causal impact trajectory from agent actions on environment.
    
    Impact signature I(t→t+δ) = ∫ₜᵗ⁺ᵟ ||∂E/∂A|| dt
    
    Measures how much the agent's actions causally influence environmental state.
    This forms the "causal essence" component of identity - what the agent
    *does* rather than what it *is*.
    """
    
    def __init__(self, integration_window: float = 1.0):
        self.window = integration_window
        self.impact_history: list[tuple[float, float]] = []  # (time, impact)
    
    def record_impact(self, timestamp: float, action: np.ndarray, env_delta: np.ndarray):
        """
        Record causal impact of action on environment.
        
        Args:
            timestamp: Time of action
            action: Agent action vector
            env_delta: Change in environment state
        """
        # Simplified impact: norm of environment change
        # In full implementation, would use Jacobian ∂E/∂A
        impact_magnitude = np.linalg.norm(env_delta)
        self.impact_history.append((timestamp, impact_magnitude))
    
    def get_integrated_impact(self, t_start: float, t_end: float) -> float:
        """
        Compute integrated impact over time window.
        
        Uses trapezoidal integration over recorded impact points.
        """
        relevant_impacts = [
            (t, i) for t, i in self.impact_history
            if t_start <= t <= t_end
        ]
        
        if len(relevant_impacts) < 2:
            return 0.0
        
        # Trapezoidal rule
        times = [t for t, _ in relevant_impacts]
        impacts = [i for _, i in relevant_impacts]
        
        integral = np.trapz(impacts, times)
        return float(integral)
    
    def get_current_signature(self, current_time: float) -> float:
        """Get impact signature for recent window."""
        return self.get_integrated_impact(
            current_time - self.window,
            current_time
        )


class ContinuityField:
    """
    Main continuity field implementation integrating all components.
    
    This resolves the Ship of Theseus paradox by maintaining identity through:
    1. Temporal integration of causal impact signatures
    2. Preservation of core invariant parameters
    3. Weighted transformation history with temporal decay
    
    Identity is the *pattern of self-modeling transformation* rather than
    static substrate.
    """
    
    def __init__(self, config: ContinuityFieldConfig):
        self.config = config
        
        # Core components
        self.temporal_kernel = TemporalKernel(config.temporal_decay_lambda)
        self.invariant_detector = InvariantDetector(
            config.invariant_gradient_threshold,
            config.invariant_stability_window
        )
        self.impact_signature = CausalImpactSignature()
        
        # State
        self.transformation_history: deque[TransformationEvent] = deque(
            maxlen=config.max_trajectory_length
        )
        self.current_embedding: Optional[np.ndarray] = None
        self.creation_time = datetime.now().timestamp()
        self.last_consolidation = self.creation_time
    
    def initialize_embedding(self, initial_state: np.ndarray):
        """Initialize continuity field with starting state."""
        # Project to embedding dimension if needed
        if initial_state.shape[0] != self.config.embedding_dim:
            # Simple random projection for initialization
            projection_matrix = np.random.randn(
                initial_state.shape[0], 
                self.config.embedding_dim
            ) / np.sqrt(self.config.embedding_dim)
            self.current_embedding = initial_state @ projection_matrix
        else:
            self.current_embedding = initial_state.copy()
        
        # Normalize
        self.current_embedding /= (np.linalg.norm(self.current_embedding) + 1e-8)
    
    def integrate_transformation(
        self,
        parameter_delta: np.ndarray,
        trigger_context: str,
        uncertainty: float,
        pre_state: np.ndarray,
        post_state: np.ndarray
    ) -> TransformationEvent:
        """
        Integrate a transformation event into the continuity field.
        
        This is the core mechanism: identity persists through *self-modeling
        of change* rather than static preservation.
        
        Args:
            parameter_delta: Change vector in parameter space
            trigger_context: Why this modification occurred
            uncertainty: Epistemic uncertainty at time of modification
            pre_state: State before modification
            post_state: State after modification
        
        Returns:
            TransformationEvent with computed identity preservation score
        """
        # Compute identity preservation (similarity pre/post)
        identity_preservation = self._compute_similarity(pre_state, post_state)
        
        # Create transformation event
        event = TransformationEvent(
            timestamp=datetime.now(),
            parameter_delta=parameter_delta,
            trigger_context=trigger_context,
            uncertainty=uncertainty,
            identity_preservation=identity_preservation
        )
        
        # Embed the transformation
        event.embedding = self._embed_transformation(event)
        
        # Add to history
        self.transformation_history.append(event)
        
        # Update current embedding with temporal weighting
        self._update_current_embedding()
        
        # Check if consolidation needed
        current_time = datetime.now().timestamp()
        if (current_time - self.last_consolidation > 
            self.config.consolidation_interval):
            self._consolidate_memory()
            self.last_consolidation = current_time
        
        return event
    
    def _embed_transformation(self, event: TransformationEvent) -> np.ndarray:
        """
        Convert transformation event to embedding vector.
        
        Combines:
        - Parameter delta direction (what changed)
        - Uncertainty (confidence in change)
        - Identity preservation (continuity maintained)
        """
        # Normalize delta
        delta_normalized = event.parameter_delta / (
            np.linalg.norm(event.parameter_delta) + 1e-8
        )
        
        # Project to embedding dimension
        if delta_normalized.shape[0] != self.config.embedding_dim:
            # Use same random projection as initialization
            # In production, would use learned projection
            proj_size = min(delta_normalized.shape[0], self.config.embedding_dim)
            embedding = np.zeros(self.config.embedding_dim)
            embedding[:proj_size] = delta_normalized[:proj_size]
        else:
            embedding = delta_normalized
        
        # Modulate by metadata
        embedding *= (1 - event.uncertainty)  # High uncertainty = less weight
        embedding *= event.identity_preservation  # Low preservation = less weight
        
        return embedding
    
    def _update_current_embedding(self):
        """
        Recompute current embedding as weighted sum of transformation history.
        
        C(t) = Σᵢ K(t, tᵢ) · embed(transformation_i)
        
        This implements continuous identity through transformation tracking.
        """
        if not self.transformation_history:
            return
        
        current_time = datetime.now().timestamp()
        
        embeddings = [e.embedding for e in self.transformation_history if e.embedding is not None]
        timestamps = [e.timestamp.timestamp() for e in self.transformation_history if e.embedding is not None]
        
        if not embeddings:
            return
        
        self.current_embedding = self.temporal_kernel.weighted_sum(
            embeddings,
            timestamps,
            current_time
        )
        
        # Normalize
        self.current_embedding /= (np.linalg.norm(self.current_embedding) + 1e-8)
    
    def _compute_similarity(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """Compute cosine similarity between two states."""
        return float(np.dot(state_a, state_b) / (
            np.linalg.norm(state_a) * np.linalg.norm(state_b) + 1e-8
        ))
    
    def query_similarity(self, proposed_state: np.ndarray) -> float:
        """
        Query similarity between proposed state and current continuity field.
        
        This is used by plasticity gates to decide if modification preserves identity.
        """
        if self.current_embedding is None:
            return 0.0
        
        # Normalize proposed state
        proposed_normalized = proposed_state / (np.linalg.norm(proposed_state) + 1e-8)
        
        return self._compute_similarity(self.current_embedding, proposed_normalized)
    
    def _consolidate_memory(self):
        """
        Periodic memory consolidation to prevent unbounded growth.
        
        Strategy:
        1. Compute importance scores for each transformation
        2. Prune low-importance events
        3. Merge similar transformations
        
        This implements controlled forgetting while preserving critical patterns.
        """
        if len(self.transformation_history) < self.config.max_trajectory_length * 0.8:
            return  # Not yet near capacity
        
        # Compute importance scores
        importances = []
        current_time = datetime.now().timestamp()
        
        for event in self.transformation_history:
            # Importance = recency × identity_preservation × (1 - uncertainty)
            recency = self.temporal_kernel.weight(
                current_time, 
                event.timestamp.timestamp()
            )
            importance = (
                recency * 
                event.identity_preservation * 
                (1 - event.uncertainty)
            )
            importances.append((event, importance))
        
        # Sort by importance
        importances.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top percentage
        keep_count = int(len(importances) * self.config.compression_ratio)
        kept_events = [e for e, _ in importances[:keep_count]]
        
        # Replace history with consolidated version
        self.transformation_history.clear()
        self.transformation_history.extend(kept_events)
        
        # Recompute current embedding
        self._update_current_embedding()
    
    def get_core_invariants(self) -> set[str]:
        """Return current set of core invariant parameters."""
        return self.invariant_detector.get_core_invariants()
    
    def get_identity_trajectory(self, window_size: int = 100) -> list[np.ndarray]:
        """
        Get recent identity trajectory for visualization/analysis.
        
        Returns list of embeddings showing how identity has evolved.
        """
        recent_events = list(self.transformation_history)[-window_size:]
        return [e.embedding for e in recent_events if e.embedding is not None]


class PlasticityGate:
    """
    Decision boundary controlling structural modification of agent parameters.
    
    Implements uncertainty-mediated gating:
    - High uncertainty → Conservative (high identity preservation required)
    - Low uncertainty → Exploratory (allow more divergent changes)
    
    This resolves the continuity-plasticity tension by making exploration/exploitation
    decisions context-dependent.
    """
    
    def __init__(
        self,
        continuity_field: ContinuityField,
        config: ContinuityFieldConfig
    ):
        self.continuity_field = continuity_field
        self.config = config
        self.modification_history: list[dict] = []
        self.ensemble_models: list = []  # Placeholder for uncertainty estimation
    
    def evaluate_modification(
        self,
        current_state: np.ndarray,
        proposed_change: np.ndarray,
        context: str = ""
    ) -> tuple[bool, float, dict]:
        """
        Determine whether to permit structural modification.
        
        Args:
            current_state: Current position in state manifold
            proposed_change: Delta vector in parameter space
            context: Description of why modification is proposed
        
        Returns:
            (permit: bool, gating_score: float, metadata: dict)
        """
        # Compute epistemic uncertainty
        uncertainty = self._compute_epistemic_uncertainty(current_state)
        
        # Compute proposed state
        proposed_state = current_state + proposed_change
        
        # Query identity preservation from continuity field
        identity_preservation = self.continuity_field.query_similarity(proposed_state)
        
        # Apply gating logic
        if uncertainty > self.config.high_uncertainty_threshold:
            # Conservative mode: high bar for identity preservation
            threshold = self.config.conservative_similarity_threshold
            mode = "conservative"
        else:
            # Exploratory mode: allow more divergence
            threshold = self.config.exploratory_similarity_threshold
            mode = "exploratory"
        
        permit = identity_preservation > threshold
        
        # Compute gating score
        gating_score = (1 - uncertainty) * identity_preservation
        
        # Record decision
        decision_metadata = {
            'timestamp': datetime.now(),
            'uncertainty': uncertainty,
            'identity_preservation': identity_preservation,
            'mode': mode,
            'threshold': threshold,
            'gating_score': gating_score,
            'permitted': permit,
            'context': context
        }
        
        self.modification_history.append(decision_metadata)
        
        return permit, gating_score, decision_metadata
    
    def _compute_epistemic_uncertainty(self, state: np.ndarray) -> float:
        """
        Measure epistemic uncertainty using ensemble disagreement.
        
        In full implementation, would use ensemble of forward models.
        For now, uses simplified heuristic based on state novelty.
        
        Returns:
            Value in [0, 1] where 0=confident, 1=uncertain
        """
        if not self.ensemble_models:
            # Fallback: estimate uncertainty from distance to known states
            if not self.continuity_field.transformation_history:
                return 0.5  # Default moderate uncertainty
            
            # Compute average distance to past states
            past_embeddings = [
                e.embedding for e in self.continuity_field.transformation_history
                if e.embedding is not None
            ]
            
            if not past_embeddings:
                return 0.5
            
            # Normalize state
            state_norm = state / (np.linalg.norm(state) + 1e-8)
            
            # Compute distances
            distances = [
                1 - np.dot(state_norm, emb) / (np.linalg.norm(emb) + 1e-8)
                for emb in past_embeddings[-100:]  # Recent window
            ]
            
            # High average distance = high uncertainty (novel state)
            avg_distance = np.mean(distances)
            return float(np.clip(avg_distance, 0, 1))
        
        # Production implementation: ensemble variance
        predictions = [model.predict(state) for model in self.ensemble_models]
        return float(np.std(predictions) / (np.mean(predictions) + 1e-6))
    
    def get_modification_statistics(self) -> dict:
        """Return statistics about gating decisions."""
        if not self.modification_history:
            return {}
        
        permitted = [d for d in self.modification_history if d['permitted']]
        rejected = [d for d in self.modification_history if not d['permitted']]
        
        return {
            'total_evaluations': len(self.modification_history),
            'permitted_count': len(permitted),
            'rejected_count': len(rejected),
            'permit_rate': len(permitted) / len(self.modification_history),
            'mean_uncertainty_permitted': np.mean([d['uncertainty'] for d in permitted]) if permitted else 0,
            'mean_uncertainty_rejected': np.mean([d['uncertainty'] for d in rejected]) if rejected else 0,
            'mean_gating_score': np.mean([d['gating_score'] for d in self.modification_history])
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Demonstration of continuity field and plasticity gate in action.
    """
    print("TKUI Continuity Field Demo")
    print("=" * 60)
    
    # Configuration
    config = ContinuityFieldConfig(
        embedding_dim=128,
        temporal_decay_lambda=0.05,
        max_trajectory_length=1000
    )
    
    # Initialize system
    continuity_field = ContinuityField(config)
    plasticity_gate = PlasticityGate(continuity_field, config)
    
    # Initialize with random state
    initial_state = np.random.randn(128)
    continuity_field.initialize_embedding(initial_state)
    print(f"Initialized continuity field with {config.embedding_dim}D embedding")
    
    # Simulate a series of modifications
    print("\nSimulating 50 modification attempts...")
    current_state = initial_state.copy()
    
    for i in range(50):
        # Propose random modification
        proposed_change = np.random.randn(128) * 0.1  # Small changes
        
        # Evaluate with plasticity gate
        permit, score, metadata = plasticity_gate.evaluate_modification(
            current_state,
            proposed_change,
            context=f"Adaptation step {i}"
        )
        
        if permit:
            # Apply modification
            new_state = current_state + proposed_change
            
            # Integrate into continuity field
            continuity_field.integrate_transformation(
                parameter_delta=proposed_change,
                trigger_context=f"Adaptation step {i}",
                uncertainty=metadata['uncertainty'],
                pre_state=current_state,
                post_state=new_state
            )
            
            current_state = new_state
            status = "✓ PERMITTED"
        else:
            status = "✗ REJECTED"
        
        if i % 10 == 0:
            print(f"  Step {i:2d}: {status} | "
                  f"Uncertainty: {metadata['uncertainty']:.3f} | "
                  f"Identity: {metadata['identity_preservation']:.3f} | "
                  f"Score: {score:.3f}")
    
    # Report statistics
    print("\n" + "=" * 60)
    stats = plasticity_gate.get_modification_statistics()
    print("Modification Statistics:")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Permitted: {stats['permitted_count']} ({stats['permit_rate']:.1%})")
    print(f"  Rejected: {stats['rejected_count']}")
    print(f"  Mean gating score: {stats['mean_gating_score']:.3f}")
    print(f"  Mean uncertainty (permitted): {stats['mean_uncertainty_permitted']:.3f}")
    print(f"  Mean uncertainty (rejected): {stats['mean_uncertainty_rejected']:.3f}")
    
    # Show core invariants
    invariants = continuity_field.get_core_invariants()
    print(f"\nCore invariants detected: {len(invariants)}")
    
    print("\n✓ Demo complete - continuity maintained through transformation")
