# Boltz-2 Core Modules Status

## Quick Stats
- **Total Core Modules**: 8
- **Ported to TTNN**: 3
- **Not Yet Ported**: 5
- **Coverage**: 37.5%

## Core Module Status

1. **InputEmbedder** ❌
   - Input processing and atom features
   - Used in: All forward passes

2. **MSAModule** ✅
   - MSA sequence processing
   - Used in: Main trunk
   - TTNN: Full implementation with all operations

3. **PairformerModule** ✅
   - Pairwise interactions
   - Used in: Main trunk
   - TTNN: Full implementation with all operations

4. **DiffusionTransformer** ✅
   - Token-level diffusion
   - Used in: Structure generation
   - TTNN: Full implementation with atom support

5. **AtomDiffusion** ❌
   - Core diffusion logic
   - Used in: Structure generation
   - Status: Only transformer part ported

6. **DiffusionConditioning** ❌
   - Diffusion setup and control
   - Used in: Structure generation

7. **DistogramModule** ❌
   - Output head for distances
   - Used in: All predictions

8. **ContactConditioning** ❌
   - Contact-based conditioning
   - Used in: Structure generation

## Next Steps
1. Port AtomDiffusion (highest impact)
2. Port InputEmbedder (needed by all)
3. Port DistogramModule (key output)
4. Port remaining diffusion components