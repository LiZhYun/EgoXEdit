"""
VACE (Video Animation and Control Engine) Model Implementation

This module implements the VACE system for advanced video editing and control in the Wan Video pipeline.
VACE enables sophisticated video manipulation capabilities including:

- Video Inpainting: Fill missing or masked regions in videos
- Video Outpainting: Extend video boundaries beyond original frame
- Object Manipulation: Add, remove, or modify objects in videos
- Style Transfer: Change video style while preserving structure
- Temporal Consistency: Maintain coherence across video frames
- Reference-guided Editing: Use reference images to guide editing

Key Architecture Components:
1. VaceWanAttentionBlock: Modified DiT blocks with skip connections for video editing
2. VaceWanModel: Main VACE model that processes editing context and generates hints
3. Patch-based Processing: Operates on video patches for efficient computation
4. Multi-layer Integration: Provides editing hints at multiple model depths

Technical Innovation:
- Uses residual connections to preserve original content while enabling edits
- Processes masked video content alongside reference information
- Integrates seamlessly with the main DiT architecture
- Supports gradient checkpointing for memory efficiency
"""

import torch
from .wan_video_dit import DiTBlock
from .utils import hash_state_dict_keys


class VaceWanAttentionBlock(DiTBlock):
    """
    VACE-enabled attention block that extends the standard DiT block.
    
    This block enhances the base DiT attention mechanism with video editing capabilities:
    - Processes editing context alongside main video content
    - Maintains residual connections to preserve original content
    - Generates editing hints that guide the main model's generation
    - Supports hierarchical editing at different model depths
    
    Architecture:
    - Inherits from DiTBlock for standard attention computation
    - Adds before_proj (block 0 only) and after_proj layers
    - Accumulates editing context across multiple blocks
    - Outputs both skip connections and refined editing hints
    
    The block operates in a residual manner:
    1. For block 0: Projects input and adds to main features
    2. For other blocks: Continues processing accumulated context
    3. Always outputs skip connection for main model integration
    """
    
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        """
        Initialize VACE attention block.
        
        Args:
            has_image_input: Whether block processes image conditioning
            dim: Hidden dimension size (1536 for 1.3B model, 5120 for 14B model)
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            eps: Layer normalization epsilon
            block_id: Block index in the VACE layer sequence
        """
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        
        # First block initializes editing context from raw VACE features
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
            
        # All blocks generate skip connections for main model integration
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        """
        Forward pass for VACE attention block.
        
        Processes editing context through attention mechanism and generates
        hints for integration with the main DiT model.
        
        Args:
            c: Accumulated VACE context from previous blocks (or initial features)
            x: Main model hidden states (for residual connection in block 0)
            context: Text conditioning from T5 encoder
            t_mod: Time modulation tensor for temporal consistency
            freqs: Positional frequency embeddings
            
        Returns:
            torch.Tensor: Stacked tensor containing [all_previous_hints, skip_connection, refined_context]
            
        Processing Flow:
        1. Block 0: Initialize editing context by projecting and adding to main features
        2. Other blocks: Extract and process accumulated context
        3. Apply standard DiT attention (self-attention + cross-attention + FFN)
        4. Generate skip connection for main model integration
        5. Accumulate all hints for hierarchical editing control
        """
        if self.block_id == 0:
            # Initialize editing context for first VACE block
            # Combines VACE features with main model state for coherent editing
            c = self.before_proj(c) + x
            all_c = []  # Start accumulating editing hints
        else:
            # Extract accumulated hints from previous blocks
            # Last element is the current context, others are skip connections
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)  # Current context for processing
        
        # Process through standard DiT attention mechanism
        # This includes: norm -> self-attention -> norm -> cross-attention -> norm -> FFN
        c = super().forward(c, context, t_mod, freqs)
        
        # Generate skip connection for main model integration
        # This hint will be added to main model features at corresponding layer
        c_skip = self.after_proj(c)
        
        # Accumulate all editing hints: [previous_hints, current_skip, current_context]
        # This creates a hierarchical set of editing signals
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    """
    VACE (Video Animation and Control Engine) Model for Advanced Video Editing.
    
    This model enables sophisticated video editing capabilities by processing
    masked video content and reference information to generate editing hints
    that guide the main DiT model's generation process.
    
    Key Capabilities:
    - Video Inpainting: Fill masked regions using surrounding context
    - Object Manipulation: Add/remove objects with reference guidance
    - Style Transfer: Apply reference style while preserving structure
    - Temporal Consistency: Maintain coherence across video frames
    - Multi-scale Editing: Provide hints at multiple model depths
    
    Architecture Overview:
    1. Patch Embedding: Converts masked video + mask into latent patches
    2. Multi-block Processing: Processes editing context through multiple attention blocks
    3. Hint Generation: Produces editing hints for integration with main model
    4. Hierarchical Control: Provides fine-grained control at different model layers
    
    Input Processing:
    - Inactive Video: Original video with masked regions set to zero
    - Reactive Video: Masked regions only (for inpainting guidance)
    - Mask Information: Spatial mask indicating editing regions
    - Reference Image: Optional reference for style/content guidance
    
    The model operates by:
    1. Encoding masked video content into patches
    2. Processing through attention blocks to understand editing intent
    3. Generating hints that modify main model behavior in masked regions
    4. Ensuring temporal consistency across video frames
    """
    
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),  # Which DiT layers receive VACE hints
        vace_in_dim=96,        # Input channels: 32 (inactive video) + 32 (reactive video) + 32 (mask)
        patch_size=(1, 2, 2),  # Temporal, Height, Width patch dimensions
        has_image_input=False, # Whether to process image conditioning
        dim=1536,              # Hidden dimension (model-specific)
        num_heads=12,          # Attention heads (model-specific)
        ffn_dim=8960,          # Feed-forward dimension (model-specific)
        eps=1e-6,              # Layer normalization epsilon
    ):
        """
        Initialize VACE model.
        
        Args:
            vace_layers: Tuple of DiT layer indices where VACE hints are applied
                        More layers = finer control but higher computation
            vace_in_dim: Input channel dimension (typically 96 = 32*3 channels)
                        32 channels each for: inactive video, reactive video, mask
            patch_size: 3D patch size for video processing (temporal, height, width)
            has_image_input: Whether model processes image conditioning
            dim: Hidden dimension size (1536 for 1.3B, 5120 for 14B model)
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            eps: Layer normalization epsilon
        """
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        
        # Create mapping from DiT layer index to VACE block index
        # This enables efficient lookup during main model forward pass
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # VACE attention blocks - one for each target DiT layer
        # These blocks process editing context and generate hints
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # Patch embedding for converting video + mask to latent patches
        # Input: [batch, 96, frames, height, width] -> [batch, dim, latent_frames, latent_h, latent_w]
        # This projection enables the model to process video editing context efficiently
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        """
        Forward pass for VACE model.
        
        Processes video editing context to generate hints that guide the main DiT model.
        The hints are applied at specific layers to enable fine-grained editing control.
        
        Args:
            x: Main model hidden states [batch, sequence_length, dim]
               Used for residual connection and dimension matching
            vace_context: List of editing context tensors
                         Each tensor: [channels, frames, height, width]
                         Typically contains: inactive video, reactive video, mask
            context: Text conditioning from T5 encoder [batch, text_tokens, text_dim]
            t_mod: Time modulation tensor for diffusion timestep
            freqs: Positional frequency embeddings for spatial/temporal positions
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            use_gradient_checkpointing_offload: Offload checkpointed activations to CPU
            
        Returns:
            List[torch.Tensor]: Editing hints for each target DiT layer
                               Each hint has shape matching the main model's hidden states
                               
        Processing Pipeline:
        1. Embed editing context into patches using 3D convolution
        2. Flatten spatial dimensions and transpose for attention processing
        3. Align sequence length with main model (pad/truncate as needed)
        4. Process through VACE attention blocks with gradient checkpointing
        5. Extract editing hints (skip connections) for main model integration
        
        Memory Optimization:
        - Gradient checkpointing reduces memory usage during training
        - CPU offloading further reduces GPU memory requirements
        - Efficient attention computation for long video sequences
        """
        # Step 1: Convert editing context to latent patches
        # Process each context tensor through patch embedding
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        
        # Step 2: Reshape for attention processing
        # Flatten spatial dimensions: [batch, dim, frames, h, w] -> [batch, dim, sequence]
        # Transpose for attention: [batch, sequence, dim]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        
        # Step 3: Handle dimension alignment with main model
        # VACE context may have different sequence length than main input
        # Need to pad or truncate to match main model's sequence length
        target_seq_len = x.shape[1] # 34320
        processed_c = []
        
        for u in c:
            if u.size(1) < target_seq_len:
                # VACE context is shorter - pad with zeros
                # This can happen with short videos or aggressive temporal compression
                padding = u.new_zeros(1, target_seq_len - u.size(1), u.size(2))
                u_processed = torch.cat([u, padding], dim=1)
            elif u.size(1) > target_seq_len:
                # VACE context is longer - truncate to match
                # This can happen with very long videos exceeding main model capacity
                u_processed = u[:, :target_seq_len, :]
            else:
                # Same length - no processing needed
                u_processed = u
            processed_c.append(u_processed)
        
        # Concatenate all processed context tensors
        c = torch.cat(processed_c)
        
        # Step 4: Define gradient checkpointing wrapper
        # This enables memory-efficient training for large models
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # Step 5: Process through VACE attention blocks
        # Each block refines the editing context and accumulates hints
        for block in self.vace_blocks:
            if use_gradient_checkpointing_offload:
                # Maximum memory efficiency: checkpoint on CPU
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                # Standard gradient checkpointing: save on GPU
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                # Standard forward pass: full memory usage
                c = block(c, x, context, t_mod, freqs) # c [1, 34320, 1536] x [1, 34320, 1536] context [1, 512, 1536]
        
        # Step 6: Extract editing hints
        # Remove the final context tensor, keeping only the skip connections
        # These hints will be added to main model features at corresponding layers
        hints = torch.unbind(c)[:-1]  # All but the last tensor are hints
        return hints
    
    @staticmethod
    def state_dict_converter():
        """
        Get state dict converter for loading pretrained VACE models.
        
        Returns:
            VaceWanModelDictConverter: Converter for different model formats
        """
        return VaceWanModelDictConverter()
    
    
class VaceWanModelDictConverter:
    """
    State dictionary converter for VACE models.
    
    Handles loading VACE models from different sources and formats:
    - CivitAI community models
    - Different model sizes (1.3B, 14B)
    - Various checkpoint formats
    
    The converter automatically detects model configuration based on
    state dict structure and provides appropriate parameter mappings.
    """
    
    def __init__(self):
        """Initialize the converter."""
        pass
    
    def from_civitai(self, state_dict):
        """
        Convert state dict from CivitAI format.
        
        Analyzes the state dict structure to determine model configuration
        and extracts VACE-specific parameters.
        
        Args:
            state_dict: Raw state dict from CivitAI checkpoint
            
        Returns:
            tuple: (filtered_state_dict, model_config)
                - filtered_state_dict: VACE parameters only
                - model_config: Configuration dictionary for model initialization
        """
        # Extract only VACE-related parameters
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("vace")}
        
        # Detect model configuration based on parameter structure
        state_dict_hash = hash_state_dict_keys(state_dict_)
        
        if state_dict_hash == '3b2726384e4f64837bdf216eea3f310d':  # VACE 14B model
            config = {
                "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35),  # 14B model layer indices
                "vace_in_dim": 96,           # Standard VACE input dimension
                "patch_size": (1, 2, 2),     # Standard patch size
                "has_image_input": False,    # Text-only conditioning
                "dim": 5120,                 # 14B model hidden dimension
                "num_heads": 40,             # 14B model attention heads
                "ffn_dim": 13824,            # 14B model FFN dimension
                "eps": 1e-06,                # Layer norm epsilon
            }
        else:
            # Default configuration for unknown models
            # User should verify these parameters match their model
            config = {}
        
        return state_dict_, config
