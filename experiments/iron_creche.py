import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from causal_compression.continuity_field import ContinuityField


# ==========================================
# 1. The World (Ground Truth Oracle)
# ==========================================
class LogicWorld:
    """
    Generates the 'Iron Creche' Triplets:
    {Original, Causal Flip (Identity), Nuisance Flip (Plasticity)}
    """

    def __init__(self):
        self.gates = ["AND", "OR", "XOR", "NAND"]
        self.states = ["High", "Low"]

    def _evaluate(self, gate, a, b):
        a_bool = True if a == "High" else False
        b_bool = True if b == "High" else False

        if gate == "AND":
            res = a_bool and b_bool
        elif gate == "OR":
            res = a_bool or b_bool
        elif gate == "XOR":
            res = a_bool != b_bool
        elif gate == "NAND":
            res = not (a_bool and b_bool)

        return "High" if res else "Low"

    def _textualize(self, gate, a, b, style="standard"):
        if style == "standard":
            return f"A:{a} B:{b} Op:{gate}"
        elif style == "verbose":
            return f"Report: A={a}, B={b} -> {gate}"
        elif style == "technical":
            return f"OP({gate}) A({a}) B({b})"

    def generate_triplet(self):
        gate = random.choice(self.gates)
        a = random.choice(self.states)
        b = random.choice(self.states)

        # 1. Original (Anchor)
        orig_text = self._textualize(gate, a, b, "standard")
        orig_y = self._evaluate(gate, a, b)

        # 2. Causal Flip (The Identity Manifold)
        # We intervene on 'Input A'. If the model misses this, it loses identity.
        a_flip = "Low" if a == "High" else "High"
        causal_text = self._textualize(gate, a_flip, b, "standard")
        causal_y = self._evaluate(gate, a_flip, b)

        # 3. Nuisance Flip (The Plasticity Manifold)
        # We change style. We want the model to compress this aggressively.
        nuisance_text = self._textualize(gate, a, b, "verbose")

        return {"x_orig": orig_text, "y_orig": orig_y, "x_causal": causal_text, "y_causal": causal_y, "x_nuis": nuisance_text}


# ==========================================
# 2. The Student (Toy Compressor Model)
# ==========================================
class ToyCompressor(nn.Module):
    """
    A simple LSTM-based compressor for demonstration.
    In production, this would be Qwen3-4B or Llama-3.
    """

    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.compressor_head = nn.Linear(hidden_dim * 2, hidden_dim)  # The bottleneck
        self.decoder = nn.Linear(hidden_dim, vocab_size)  # Reconstructs key info

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder(embedded)
        # For bidirectional LSTM, hidden is (num_layers * num_directions, batch, hidden_size)
        # We need to concatenate the forward and backward hidden states for the last layer.
        # hidden[0] is forward, hidden[1] is backward for the last layer if num_layers=1
        hidden_combined = torch.cat((hidden[0], hidden[1]), dim=1)
        # This 'compressed' vector is what we store in the Conceptual Lattice
        compressed = self.compressor_head(hidden_combined)
        prediction = self.decoder(compressed)
        return prediction, compressed


# ==========================================
# 3. The Geometric Projection (RAA Core)
# ==========================================
def project_gradients(model, grad_identity, grad_plasticity):
    """
    Implements: P_mutable = I - P_identity
    Ensures compression updates are orthogonal to causal logic.
    """
    safe_grads = []

    # Iterate through every parameter (weight) in the model
    for i, (p_name, param) in enumerate(model.named_parameters()):
        if param.grad is None:
            continue

        g_ident = grad_identity[i]  # Gradient from Causal Flip
        g_plast = grad_plasticity[i]  # Gradient from Nuisance Compression

        # Flatten for dot product
        g_i_flat = g_ident.view(-1)
        g_p_flat = g_plast.view(-1)

        # Calculate Projection: (g_p . g_i) / (g_i . g_i)
        # How much of the compression update lies on the identity manifold?
        dot_prod = torch.dot(g_p_flat, g_i_flat)
        norm_sq = torch.dot(g_i_flat, g_i_flat) + 1e-8
        scalar = dot_prod / norm_sq

        if i == 0 and random.random() < 0.1:  # Print scalar for first param occasionally
            print(f"DEBUG: Projection scalar (layer 0) = {scalar.item():.4f}")

        # Subtract the component that would hurt causal logic
        # orthogonal_update = g_plasticity - scalar * g_identity
        ortho_grad = g_plast - (scalar * g_ident)

        # The final gradient mixes the purely causal update with the safe compression update
        # We enforce Identity (causal accuracy) + Plasticity (orthogonal compression)
        final_grad = g_ident + ortho_grad

        safe_grads.append(final_grad)

    return safe_grads


# ==========================================
# 4. The Training Loop
# ==========================================
def run_iron_creche_training(steps=100, use_projection=True):
    world = LogicWorld()
    # Mock vocab setup for demo
    vocab = {char: i for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ_ :.=0123456789abcdefghijklmnopqrstuvwxyz")}
    vocab_size = len(vocab)

    # Initialize Continuity Field (Identity Manifold)
    hidden_dim = 64
    continuity_field = ContinuityField(embedding_dim=hidden_dim, k_neighbors=5)

    model = ToyCompressor(vocab_size, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Helper to vectorize text
    def tokenize(text):
        return torch.tensor([[vocab.get(c, 0) for c in text]], dtype=torch.long)

    print(f"--- Starting Training (Projection={use_projection}) ({steps} steps) ---")
    torch.manual_seed(42)

    for step in range(steps):
        triplet = world.generate_triplet()

        # 1. Compute Identity Gradient (Causal Flip)
        # We want the model to distinguish x_orig from x_causal
        optimizer.zero_grad()

        # Forward pass on Original
        t_orig = tokenize(triplet["x_orig"])
        _, z_orig = model(t_orig)

        # Forward pass on Causal Flip (Intervention)
        t_causal = tokenize(triplet["x_causal"])
        _, z_causal = model(t_causal)

        # Add Identity State to Manifold (Simulating 'Sleep' or Consolidation)
        # In a real system, we'd only add stable/verified states
        if step % 10 == 0:
            continuity_field.add_anchor(z_orig.detach().numpy().flatten())

        # Measure Drift from Identity Manifold
        # If the model is preserving identity, z_causal should be 'close' to the manifold of z_orig
        # Note: In this specific toy setup, x_causal is the SAME identity as x_orig, just flipped.
        # So z_causal SHOULD be on the manifold defined by z_orig states.
        try:
            drift_metric = continuity_field.get_drift_metric(z_causal.detach().numpy().flatten())
        except ValueError:
            drift_metric = 0.0  # Field empty

        if step == 0:
            print(f"DEBUG: x_orig='{triplet['x_orig']}'")
            print(f"DEBUG: x_causal='{triplet['x_causal']}'")
            print(f"DEBUG: t_orig={t_orig}")
            print(f"DEBUG: t_causal={t_causal}")
            print(f"DEBUG: z_orig norm={torch.norm(z_orig).item()}")
            print(f"DEBUG: z_causal norm={torch.norm(z_causal).item()}")
            print(f"DEBUG: diff norm={torch.norm(z_orig - z_causal).item()}")
            print(f"DEBUG: Manifold Drift={drift_metric}")

        # LOSS 1: ISS (Intervention Sensitivity Score)
        # We want z_orig and z_causal to be DISTANT (maximize difference)
        # We minimize negative distance
        dist = torch.norm(z_orig - z_causal)
        loss_identity = -dist  # We want distance to increase

        loss_identity.backward(retain_graph=True)
        # Capture gradients for P_identity
        grad_identity = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]

        if step < 5:
            g_norm = sum(g.norm() for g in grad_identity)
            print(f"DEBUG: Step {step} grad_identity norm={g_norm.item():.4e}")

        # 2. Compute Plasticity Gradient (Compression)
        # We want to minimize length/entropy of the Nuisance input
        optimizer.zero_grad()

        t_nuis = tokenize(triplet["x_nuis"])
        pred_nuis, z_nuis = model(t_nuis)

        # LOSS 2: Invariance/Compression
        # We want z_nuis to be CLOSE to z_orig (ignore syntax changes)
        # AND we want the vector norm to be small (compression)
        dist_inv = torch.norm(z_orig - z_nuis)
        sparsity = torch.norm(z_nuis)
        loss_plasticity = dist_inv + (0.5 * sparsity)

        loss_plasticity.backward()
        # Capture gradients for P_mutable
        grad_plasticity = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]

        # 3. Apply Projected Plasticity
        if use_projection:
            # Calculate the safe update
            final_grads = project_gradients(model, grad_identity, grad_plasticity)
        else:
            # Standard Update (Naive Multi-task)
            # Just add gradients: Identity + Plasticity
            final_grads = [g_i + g_p for g_i, g_p in zip(grad_identity, grad_plasticity)]

        # Apply safely projected gradients to the model
        optimizer.zero_grad()
        for param, safe_grad in zip(model.parameters(), final_grads):
            if param.grad is not None:
                param.grad = safe_grad

        optimizer.step()

        if step < 10 or step % 50 == 0:
            print(f"Step {step}: Identity Dist={dist.item():.4e} (Target: High) | Invariance Dist={dist_inv.item():.4e} (Target: Low) | Manifold Drift={drift_metric:.4f}")

    print("--- Training Complete ---")
    return model


if __name__ == "__main__":
    print("\n=== Experiment 1: With Projection (Safe) ===")
    run_iron_creche_training(steps=200, use_projection=True)

    print("\n=== Experiment 2: Without Projection (Unsafe) ===")
    run_iron_creche_training(steps=200, use_projection=False)
