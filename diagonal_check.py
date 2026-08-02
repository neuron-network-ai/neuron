"""
diagonal_check.py — Does the cube diagonal idea work on real weights?
======================================================================
Author: NEURON Labs / NEURON Network

Tests whether real Qwen2.5-1.5B weight matrices have diagonal structure
that allows corner sampling to predict center activation patterns.

If YES: the cube diagonal predictor is worth building
If NO:  the weight structure does not support this approach

RUN ON YOUR MACHINE:
  python diagonal_check.py --model-dir ./model_slice/
  or
  python diagonal_check.py --model-dir ~/.cache/huggingface/hub/.../

What it checks:
  1. Diagonal magnitude pattern across the full matrix
  2. Whether corner diagonals correlate with center diagonals
  3. Whether diagonal magnitude predicts row activation magnitude
  4. Visualization of the tile structure

Output: pass/fail verdict with correlation scores
"""

import sys
import math
import os

def load_weight(model_dir, layer=10, name="gate_proj"):
    """Load one real weight matrix from safetensors"""
    try:
        from safetensors import safe_open
    except ImportError:
        print("Installing safetensors...")
        os.system("pip install safetensors -q")
        from safetensors import safe_open

    # Try different path patterns
    patterns = [
        f"{model_dir}/model.safetensors",
        f"{model_dir}/model-00001-of-*.safetensors",
    ]
    
    import glob
    sf_file = None
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            sf_file = matches[0]
            break
    
    if not sf_file:
        # Try finding any safetensors file
        files = glob.glob(f"{model_dir}/**/*.safetensors", recursive=True)
        if files:
            sf_file = files[0]
    
    if not sf_file:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    
    print(f"Loading from: {sf_file}")
    key = f"model.layers.{layer}.mlp.{name}.weight"
    
    with safe_open(sf_file, framework="pt") as f:
        keys = list(f.keys())
        # Find matching key
        matching = [k for k in keys if name in k and f"layers.{layer}" in k]
        if not matching:
            print(f"Key {key} not found. Available keys with '{name}':")
            for k in keys:
                if name in k:
                    print(f"  {k}")
            # Use first available mlp weight
            mlp_keys = [k for k in keys if 'mlp' in k and 'weight' in k]
            if mlp_keys:
                key = mlp_keys[0]
                print(f"Using: {key}")
            else:
                raise KeyError(f"No mlp weight found")
        else:
            key = matching[0]
        
        W = f.get_tensor(key).float().numpy()
        print(f"Loaded: {key}  shape={W.shape}")
        return W, key

def check_diagonal_structure(W, label=""):
    """
    Core analysis: does the diagonal structure predict row activation?
    
    Returns correlation score 0-1
    Higher = diagonal structure is real and predictive
    """
    od, id_ = W.shape
    print(f"\n{'='*60}")
    print(f"Matrix: {label}  [{od} × {id_}]")
    print(f"{'='*60}")
    
    # ─────────────────────────────────────────────
    # 1. Row magnitude (what we want to predict)
    # ─────────────────────────────────────────────
    row_mag = []
    for i in range(od):
        mag = sum(abs(float(W[i,j])) for j in range(id_)) / id_
        row_mag.append(mag)
    
    mean_row_mag = sum(row_mag) / len(row_mag)
    std_row_mag  = math.sqrt(sum((x-mean_row_mag)**2 
                                  for x in row_mag) / len(row_mag))
    
    print(f"\nRow magnitude stats:")
    print(f"  mean = {mean_row_mag:.6f}")
    print(f"  std  = {std_row_mag:.6f}")
    print(f"  max  = {max(row_mag):.6f}")
    print(f"  min  = {min(row_mag):.6f}")
    
    # What fraction of rows are "spikes" (above mean)?
    spike_rows = sum(1 for m in row_mag if m > mean_row_mag * 1.5)
    print(f"  spike rows (>1.5x mean): {spike_rows}/{od} = {spike_rows/od*100:.1f}%")
    
    # ─────────────────────────────────────────────
    # 2. Diagonal magnitude in each corner
    # ─────────────────────────────────────────────
    # Corner size: sample a small window from each corner
    corner_size = min(128, od//4, id_//4)
    corner_size = max(32, corner_size)
    
    print(f"\nDiagonal analysis (corner_size={corner_size}):")
    
    corners = {
        'TL': (0,          0),
        'TR': (0,          id_-corner_size),
        'BL': (od-corner_size, 0),
        'BR': (od-corner_size, id_-corner_size),
    }
    
    corner_diag_mags = {}
    for name, (r0, c0) in corners.items():
        diag_mag = []
        for k in range(corner_size):
            ri = r0 + k
            ci = c0 + k
            if ri < od and ci < id_:
                diag_mag.append(abs(float(W[ri, ci])))
        corner_diag_mags[name] = diag_mag
        mean_diag = sum(diag_mag)/len(diag_mag) if diag_mag else 0
        print(f"  Corner {name} diagonal mean: {mean_diag:.6f}")
    
    # ─────────────────────────────────────────────
    # 3. Center diagonal
    # ─────────────────────────────────────────────
    center_r = od // 2 - corner_size // 2
    center_c = id_ // 2 - corner_size // 2
    center_diag = []
    for k in range(corner_size):
        ri = center_r + k
        ci = center_c + k
        if ri < od and ci < id_:
            center_diag.append(abs(float(W[ri, ci])))
    
    center_mean = sum(center_diag)/len(center_diag) if center_diag else 0
    print(f"  Center diagonal mean:      {center_mean:.6f}")
    
    # ─────────────────────────────────────────────
    # 4. Cube diagonal — corner to opposite corner
    # ─────────────────────────────────────────────
    print(f"\nCube diagonal (TL corner → BR corner):")
    
    # Sample points along the diagonal path from TL to BR
    n_samples = 20
    cube_diag_mags = []
    for s in range(n_samples):
        frac = s / (n_samples - 1)
        # Row position along diagonal
        ri = int(frac * (od - 1))
        # Column position along diagonal  
        ci = int(frac * (id_ - 1))
        # Sample a small window around this point
        window_size = 8
        window_mag = 0
        count = 0
        for dr in range(-window_size//2, window_size//2):
            for dc in range(-window_size//2, window_size//2):
                r = ri + dr
                c = ci + dc
                if 0 <= r < od and 0 <= c < id_:
                    window_mag += abs(float(W[r, c]))
                    count += 1
        if count > 0:
            cube_diag_mags.append(window_mag / count)
    
    # Does cube diagonal magnitude correlate with row magnitude
    # at same position?
    correlations = []
    for s in range(n_samples):
        frac = s / (n_samples - 1)
        ri = int(frac * (od - 1))
        diag_val = cube_diag_mags[s]
        row_val   = row_mag[ri]
        correlations.append((diag_val, row_val))
    
    # Pearson correlation between diagonal magnitude and row magnitude
    diag_vals = [c[0] for c in correlations]
    row_vals  = [c[1] for c in correlations]
    
    mean_d = sum(diag_vals)/len(diag_vals)
    mean_r = sum(row_vals)/len(row_vals)
    
    num = sum((d-mean_d)*(r-mean_r) 
              for d,r in zip(diag_vals,row_vals))
    den_d = math.sqrt(sum((d-mean_d)**2 for d in diag_vals))
    den_r = math.sqrt(sum((r-mean_r)**2 for r in row_vals))
    
    if den_d > 1e-10 and den_r > 1e-10:
        correlation = num / (den_d * den_r)
    else:
        correlation = 0.0
    
    print(f"  Cube diagonal vs row magnitude correlation: {correlation:.4f}")
    
    # ─────────────────────────────────────────────
    # 5. Anti-diagonal (TR to BL)
    # ─────────────────────────────────────────────
    anti_diag_mags = []
    for s in range(n_samples):
        frac = s / (n_samples - 1)
        ri = int(frac * (od - 1))
        ci = int((1-frac) * (id_ - 1))  # reverse column
        window_mag = 0; count = 0
        for dr in range(-4, 4):
            for dc in range(-4, 4):
                r = ri+dr; c = ci+dc
                if 0<=r<od and 0<=c<id_:
                    window_mag += abs(float(W[r,c])); count+=1
        if count > 0:
            anti_diag_mags.append(window_mag/count)
    
    anti_diag_vals = anti_diag_mags
    mean_a = sum(anti_diag_vals)/len(anti_diag_vals)
    
    num_a = sum((a-mean_a)*(r-mean_r) 
                for a,r in zip(anti_diag_vals,row_vals))
    den_a = math.sqrt(sum((a-mean_a)**2 for a in anti_diag_vals))
    
    if den_a > 1e-10 and den_r > 1e-10:
        anti_corr = num_a / (den_a * den_r)
    else:
        anti_corr = 0.0
    
    print(f"  Anti-diagonal vs row magnitude correlation: {anti_corr:.4f}")
    
    # ─────────────────────────────────────────────
    # 6. Can corner predict center?
    # ─────────────────────────────────────────────
    tl_mean = sum(corner_diag_mags['TL'])/len(corner_diag_mags['TL'])
    br_mean = sum(corner_diag_mags['BR'])/len(corner_diag_mags['BR'])
    pred_center = (tl_mean + br_mean) / 2
    
    center_err = abs(pred_center - center_mean) / max(center_mean, 1e-10)
    print(f"\nCorner prediction of center diagonal:")
    print(f"  TL+BR average:  {pred_center:.6f}")
    print(f"  Actual center:  {center_mean:.6f}")
    print(f"  Relative error: {center_err:.4f} ({center_err*100:.1f}%)")
    
    # ─────────────────────────────────────────────
    # 7. VERDICT
    # ─────────────────────────────────────────────
    print(f"\nVERDICT for {label}:")
    
    max_corr = max(abs(correlation), abs(anti_corr))
    
    if max_corr > 0.5:
        print(f"  ✅ STRONG diagonal structure (corr={max_corr:.3f})")
        print(f"     Cube diagonal predictor likely to work")
        verdict = "STRONG"
    elif max_corr > 0.2:
        print(f"  ⚠️  WEAK diagonal structure (corr={max_corr:.3f})")
        print(f"     Cube diagonal may help but unreliable")
        verdict = "WEAK"
    else:
        print(f"  ❌ NO diagonal structure (corr={max_corr:.3f})")
        print(f"     Weights are too uniform for diagonal prediction")
        verdict = "NONE"
    
    if center_err < 0.1:
        print(f"  ✅ Corner predicts center well ({center_err*100:.1f}% error)")
    else:
        print(f"  ❌ Corner does not predict center ({center_err*100:.1f}% error)")
    
    return {
        'label':       label,
        'shape':       (od, id_),
        'correlation': correlation,
        'anti_corr':   anti_corr,
        'center_err':  center_err,
        'spike_frac':  spike_rows/od,
        'verdict':     verdict,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='.',
                        help='Directory containing model.safetensors')
    parser.add_argument('--layer', type=int, default=10,
                        help='Which transformer layer to test')
    args = parser.parse_args()
    
    print("="*60)
    print("Diagonal Structure Check for NeuronScript Cube Predictor")
    print("Testing the cube diagonal idea on real Qwen2.5 weights")
    print("="*60)
    
    results = []
    
    # Test multiple weight matrices
    test_cases = [
        (args.layer, "gate_proj",  "FFN gate"),
        (args.layer, "down_proj",  "FFN down"),
        (args.layer, "up_proj",    "FFN up"),
        (args.layer, "q_proj",     "Attention Q"),
        (args.layer, "k_proj",     "Attention K"),
    ]
    
    for layer, name, label in test_cases:
        try:
            W, key = load_weight(args.model_dir, layer, name)
            result = check_diagonal_structure(W, f"{label} [{key}]")
            results.append(result)
        except Exception as e:
            print(f"\nSkipping {label}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Should we build the cube diagonal predictor?")
    print(f"{'='*60}")
    
    if not results:
        print("No weights could be loaded. Check --model-dir path.")
        return
    
    strong = sum(1 for r in results if r['verdict'] == 'STRONG')
    weak   = sum(1 for r in results if r['verdict'] == 'WEAK')
    none_  = sum(1 for r in results if r['verdict'] == 'NONE')
    
    print(f"\n  Strong diagonal structure: {strong}/{len(results)} matrices")
    print(f"  Weak diagonal structure:   {weak}/{len(results)} matrices")
    print(f"  No diagonal structure:     {none_}/{len(results)} matrices")
    
    avg_corr = sum(max(abs(r['correlation']),
                       abs(r['anti_corr'])) for r in results) / len(results)
    avg_spike = sum(r['spike_frac'] for r in results) / len(results)
    
    print(f"\n  Average diagonal correlation: {avg_corr:.4f}")
    print(f"  Average spike fraction:       {avg_spike*100:.1f}%")
    
    print()
    if strong >= len(results) // 2:
        print("  ✅ BUILD IT")
        print("  Diagonal structure is real in real transformer weights")
        print("  Cube diagonal predictor is worth implementing")
        print("  Expected to predict and skip 30-50% of computation")
    elif strong + weak >= len(results) // 2:
        print("  ⚠️  BUILD WITH CAUTION")
        print("  Some diagonal structure exists but inconsistent")
        print("  Start with conservative skip threshold (10-20%)")
        print("  Validate quality carefully on each layer type")
    else:
        print("  ❌ DO NOT BUILD YET")
        print("  Diagonal structure not found in these weight matrices")
        print("  The cube idea needs a different mathematical foundation")
        print("  OR these specific weights lack the expected structure")
        print()
        print("  Possible next step: check if structure exists in")
        print("  LARGER models (7B, 70B) where weights may be more")
        print("  structured due to more training data and capacity")
    
    print()
    print("Run this script on your machine with real Qwen2.5 weights:")
    print("  python diagonal_check.py --model-dir /path/to/model/")

if __name__ == '__main__':
    main()
