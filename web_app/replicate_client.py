"""
Replicate API client for PharmaLink GNN.

Provides:
  1. generate_molecule_image()     – SDXL concept art of a drug candidate
  2. generate_fuzzy_visual_image() – SDXL artistic view of the fuzzy neural network
  3. generate_molecules_text()     – LLM-based SMILES generation (Llama-3 on Replicate)

Configure via:
  - Environment variable  REPLICATE_API_TOKEN
  - Or call configure(token) at runtime (mirroring gemini_client pattern)
"""

import os

# ─── SDXL model version on Replicate ──────────────────────────────────────────
_SDXL_VERSION = (
    "stability-ai/sdxl:"
    "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
)

# ─── Llama-3 8B Instruct on Replicate (for molecule SMILES generation) ───────
_LLM_VERSION = (
    "meta/meta-llama-3-8b-instruct"
)

_token: str | None = None


def configure(token: str):
    """Set the Replicate API token at runtime."""
    global _token
    _token = token.strip()
    os.environ['REPLICATE_API_TOKEN'] = _token


def is_ready() -> bool:
    return bool(_token) or bool(os.environ.get('REPLICATE_API_TOKEN'))


def _client():
    """Return a replicate client, raising if unavailable."""
    if not is_ready():
        raise RuntimeError(
            "Replicate chưa được cấu hình. Vui lòng thêm REPLICATE_API_TOKEN."
        )
    try:
        import replicate
    except ImportError:
        raise RuntimeError(
            "Thư viện 'replicate' chưa được cài. Chạy: pip install replicate"
        )
    return replicate


# ════════════════════════════════════════════════════════════════
#  1. Drug molecule concept-art image
# ════════════════════════════════════════════════════════════════

def generate_molecule_image(disease_name: str,
                            molecule_name: str,
                            smiles: str = '') -> str:
    """
    Generate a scientific concept image of a drug candidate via Stable Diffusion XL.

    Parameters
    ----------
    disease_name   : target disease (used in the SDXL prompt)
    molecule_name  : candidate drug name / label
    smiles         : SMILES string (optionally included in prompt)

    Returns
    -------
    URL string of the generated image (first output from SDXL).
    Empty string on failure.
    """
    rep = _client()

    smiles_hint = f", inspired by SMILES: {smiles[:40]}" if smiles else ""
    prompt = (
        f"A photorealistic scientific illustration of a pharmaceutical drug molecule "
        f"targeting {disease_name}, compound named {molecule_name}{smiles_hint}, "
        f"glowing molecular structure with atoms and chemical bonds, "
        f"bioluminescent colors, dark laboratory background, "
        f"ultra-detailed scientific art, 4K render"
    )
    negative = (
        "blurry, text, watermark, cartoon, anime, low quality, "
        "extra limbs, deformed"
    )

    try:
        output = rep.run(
            _SDXL_VERSION,
            input={
                "prompt": prompt,
                "negative_prompt": negative,
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "num_outputs": 1,
            },
        )
        if output:
            return str(output[0])
    except Exception as e:
        print(f"[Replicate] generate_molecule_image error: {e}")
    return ""


# ════════════════════════════════════════════════════════════════
#  2. Fuzzy neural network visualisation image
# ════════════════════════════════════════════════════════════════

def generate_fuzzy_visual_image(drug_name: str = '',
                                disease_name: str = '',
                                n_rules: int = 32) -> str:
    """
    Generate an artistic image representing the fuzzy neural network
    processing a drug–disease pair.

    Returns
    -------
    URL string of the generated image, or empty string on failure.
    """
    rep = _client()

    context = ""
    if drug_name and disease_name:
        context = f"analysing drug '{drug_name}' and disease '{disease_name}', "

    prompt = (
        f"Abstract 3D scientific visualisation of a fuzzy neural network {context}"
        f"with {n_rules} glowing Gaussian membership function curves floating in space, "
        f"soft gradient blue-purple-cyan colour palette, "
        f"interconnected neural nodes with fuzzy blurred edges representing uncertainty, "
        f"Perlin noise field background, volumetric light rays, "
        f"cinematic render, 4K ultra detail, no text"
    )
    negative = "blurry, low quality, text, watermark, realistic people, cartoon"

    try:
        output = rep.run(
            _SDXL_VERSION,
            input={
                "prompt": prompt,
                "negative_prompt": negative,
                "width": 1216,
                "height": 832,
                "num_inference_steps": 25,
                "guidance_scale": 8.0,
                "num_outputs": 1,
            },
        )
        if output:
            return str(output[0])
    except Exception as e:
        print(f"[Replicate] generate_fuzzy_visual_image error: {e}")
    return ""


# ════════════════════════════════════════════════════════════════
#  3. LLM-based drug molecule SMILES generation
# ════════════════════════════════════════════════════════════════

def generate_molecules_llm(disease_name: str,
                           known_drugs: list[str],
                           n_candidates: int = 6) -> list[dict]:
    """
    Ask Llama-3-8B on Replicate to propose novel drug-candidate SMILES
    that target the given disease, inspired by the known drugs list.

    Returns a list of dicts with keys: name, smiles, rationale, strategy.
    Falls back to empty list on any error.
    """
    rep = _client()

    known_str = ", ".join(known_drugs[:5]) if known_drugs else "không có"

    system_prompt = (
        "You are a medicinal chemistry expert. Respond ONLY with a valid JSON array. "
        "No markdown, no explanation outside the array."
    )
    user_prompt = (
        f"Generate {n_candidates} novel drug-candidate SMILES strings targeting the disease: {disease_name}.\n"
        f"Known drugs for this disease (use as inspiration): {known_str}.\n\n"
        "For each candidate output a JSON object with exactly these keys:\n"
        '  "name": short descriptive name,\n'
        '  "smiles": valid SMILES string,\n'
        '  "strategy": one of ["Fragment Addition", "Scaffold Hybridization", "Bioisostere Replacement"],\n'
        '  "rationale": one sentence in Vietnamese explaining why this molecule may work.\n\n'
        f"Output a JSON array of {n_candidates} such objects."
    )

    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
    full_prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
    full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    try:
        output_parts = []
        for chunk in rep.run(
            _LLM_VERSION,
            input={
                "prompt": full_prompt,
                "max_new_tokens": 1500,
                "temperature": 0.6,
                "top_p": 0.9,
            },
        ):
            output_parts.append(str(chunk))

        raw = "".join(output_parts).strip()

        # Strip any markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        import json
        candidates = json.loads(raw)
        if isinstance(candidates, list):
            return candidates[:n_candidates]
    except Exception as e:
        print(f"[Replicate] generate_molecules_llm error: {e}")

    return []
