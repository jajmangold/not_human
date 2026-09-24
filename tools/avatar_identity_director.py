"""Avatar-identity director — a proof-of-concept, spike-scoped nothuman Stage-0
pipeline: structured character spec -> deterministic prompt compiler -> optional
LLM-drafted creative notes -> fail-closed hairstyle/framing checker -> zimage-fni8
render -> ready for IDOL reconstruction.

Modeled on content-factory's infra#629 album-cover-director (draft -> check ->
revise -> render LangGraph shape) and its `character.design` job kind's structured
`inputs.character = {id, role, silhouette, materials, palette}` spec -- but
reimplemented natively here rather than imported, since content-factory and
nothuman are separately governed repos with separate licensing postures (the same
reason musetalk-volta is reused as reference evidence, never as a runtime
dependency, for nothuman's own MuseTalk/LivePortrait adapters).

Key difference from the album-cover-director: identity fields here are mostly
enumerable (hairstyle, top, bottom, shoes) rather than open creative prose, so the
"check" step's most important job is a hairstyle allow-list encoded from this
session's own empirical finding (2026-09-10): short-cut and tied-back hairstyles
reconstruct cleanly through IDOL's Gaussian representation; long flowing hair reads
as "plastic". That constraint has authority 1.0 -- it is enforced on the FINAL
compiled prompt regardless of what an optional LLM-drafted `notes` field contributes,
mirroring nothuman's own
`resolved = commanded + permitted_additive_residuals + (1-authority)*coherence_residual`
philosophy, just applied to prompt text instead of motion.

NOT governed nothuman code. Spike only -- run manually, evidence-only, same pattern
as this session's other IDOL spike scripts.
"""
from __future__ import annotations

import base64
import dataclasses
import json
import re
import urllib.request
from pathlib import Path
from typing import Optional

ZIMAGE_ENDPOINT = "http://127.0.0.1:9000/generate"
GEMMA_ENDPOINT = "http://127.0.0.1:8031/v1/chat/completions"

# Fail-closed allow-list: only hairstyles empirically confirmed (or clearly
# analogous to a confirmed style) to reconstruct cleanly through IDOL. Each entry
# maps to the exact descriptive phrase used in the validated prompt template.
# Add to this list only after a new style is turntable-verified, the same way
# pixie_bob and slicked_ponytail were verified this session.
# Each phrase takes a {color} slot -- keeps grammar correct regardless of
# hair_color instead of prepending it awkwardly in front of an article.
HAIRSTYLE_ALLOWLIST: dict[str, str] = {
    "pixie_bob": "a short {color} pixie bob haircut, hair neatly tucked behind the ears",
    "slicked_ponytail": "{color} hair pulled back tightly into a neat ponytail, no loose strands",
    "buzz_cut": "a very short {color} buzz cut, hair cropped close to the scalp all over",
    "low_bun": "{color} hair pulled back smoothly into a low neat bun at the nape of the neck",
    "completely_bald": "a completely bald, smooth shaved head, no hair at all",
}

# Denylist scanned against the FINAL compiled prompt text, independent of source
# (deterministic template or LLM-drafted notes). This is the hard-authority gate --
# nothing downstream of this function may reintroduce banned vocabulary.
LONG_HAIR_DENYLIST = [
    "long hair", "flowing hair", "flowing locks", "wavy hair down",
    "loose hair", "hair down her back", "hair past her shoulders",
    "hair past his shoulders", "cascading hair", "hair blowing",
    "wig", "extensions", "windswept hair",
]

REQUIRED_FRAMING_TERMS = ["full body", "standing", "background"]
MIN_WORDS, MAX_WORDS = 25, 120


@dataclasses.dataclass
class CharacterSpec:
    id: str
    role: str
    gender_presentation: str  # free text, e.g. "woman", "man", "androgynous person"
    hairstyle: str  # MUST be a key in HAIRSTYLE_ALLOWLIST
    hair_color: str
    top: str
    bottom: str
    shoes: str
    seed: int
    notes: Optional[str] = None  # optional free-text creative direction


class CheckFailure(Exception):
    pass


def compile_base_prompt(spec: CharacterSpec) -> str:
    if spec.hairstyle not in HAIRSTYLE_ALLOWLIST:
        raise CheckFailure(
            f"hairstyle {spec.hairstyle!r} is not in the allow-list "
            f"{sorted(HAIRSTYLE_ALLOWLIST)} -- only turntable-verified styles are permitted"
        )
    hair_phrase = HAIRSTYLE_ALLOWLIST[spec.hairstyle].format(color=spec.hair_color)
    return (
        f"full body studio photo of a young {spec.gender_presentation} with "
        f"{hair_phrase}, standing straight facing the camera, "
        f"arms slightly away from their sides, plain pure white seamless background, "
        f"even soft studio lighting, wearing {spec.top} and {spec.bottom} and "
        f"{spec.shoes}, sharp focus, photorealistic stock photo"
    )


def draft_notes_clause(notes: str) -> str:
    """Ask gemma4-26b to turn free-text creative notes into a short visual
    descriptor clause. This is the only open-ended, LLM-drafted part of the
    prompt -- everything else (hair, framing, clothing) is deterministic.
    Failure here just means no extra clause is added (fail-open on the
    non-safety-critical path, fail-closed on hair happens separately below)."""
    body = json.dumps({
        "model": "gemma4-26b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Convert the given character personality/vibe notes into ONE short "
                    "visual descriptor clause (under 15 words) suitable for appending to a "
                    "photorealistic studio-photo prompt. Describe only visible things: "
                    "expression, posture, accessories, color mood. Never mention hair length "
                    "or hairstyle -- that is fixed separately and must not be touched. "
                    "Reply with ONLY the clause, no quotes, no preamble."
                ),
            },
            {"role": "user", "content": notes},
        ],
        "max_tokens": 60,
        "temperature": 0.4,
    }).encode()
    req = urllib.request.Request(
        GEMMA_ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def quality_issues(prompt: str) -> list[str]:
    """Deterministic checker -- the fail-closed gate. Runs on the FINAL compiled
    prompt regardless of whether an LLM contributed to it."""
    issues = []
    lower = prompt.lower()
    for banned in LONG_HAIR_DENYLIST:
        if banned in lower:
            issues.append(f"denylisted hair phrase present: {banned!r}")
    for required in REQUIRED_FRAMING_TERMS:
        if required not in lower:
            issues.append(f"missing required framing term: {required!r}")
    word_count = len(prompt.split())
    if not (MIN_WORDS <= word_count <= MAX_WORDS):
        issues.append(f"word count {word_count} outside [{MIN_WORDS}, {MAX_WORDS}]")
    return issues


def direct(spec: CharacterSpec) -> dict:
    """draft -> check -> (revise by dropping the LLM clause) -> compliant prompt."""
    base_prompt = compile_base_prompt(spec)
    prompt = base_prompt
    llm_clause = None
    revisions = 0

    if spec.notes:
        try:
            llm_clause = draft_notes_clause(spec.notes)
            candidate = f"{base_prompt}, {llm_clause}"
            issues = quality_issues(candidate)
            if not issues:
                prompt = candidate
            else:
                # revise: the LLM clause violated something -- drop it and fall
                # back to the deterministic-only prompt. Hair/framing authority
                # is 1.0: an LLM contribution can never override it.
                revisions += 1
        except Exception as exc:  # noqa: BLE001 -- draft step is best-effort
            revisions += 1
            llm_clause = f"<draft failed, dropped: {exc}>"

    final_issues = quality_issues(prompt)
    if final_issues:
        raise CheckFailure(f"final prompt failed checks: {final_issues}")

    return {
        "spec": dataclasses.asdict(spec),
        "base_prompt": base_prompt,
        "llm_clause": llm_clause,
        "final_prompt": prompt,
        "revisions": revisions,
        "checks_passed": True,
    }


def render(prompt: str, seed: int, width: int = 768, height: int = 1152) -> bytes:
    body = json.dumps({
        "width": width, "height": height, "prompt": prompt, "seed": seed,
    }).encode()
    req = urllib.request.Request(
        ZIMAGE_ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    if "error" in data:
        raise RuntimeError(f"zimage-fni8 generate failed: {data['error']}")
    return base64.b64decode(data["image_png"]), data.get("provenance", {}), data.get("timing", {})


def main():
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # A genuinely new test character (male presentation, buzz cut -- not one of
    # today's three prior samples) to check the pipeline generalizes rather than
    # just replaying known-good prompts.
    spec = CharacterSpec(
        id="test-buzzcut-01",
        role="assistant avatar",
        gender_presentation="man",
        hairstyle="buzz_cut",
        hair_color="dark brown",
        top="a fitted navy crew-neck t-shirt",
        bottom="charcoal grey chinos",
        shoes="white leather sneakers",
        seed=5001,
        notes="calm, approachable, quietly confident",
    )

    result = direct(spec)
    print(json.dumps({k: v for k, v in result.items() if k != "spec"}, indent=2))

    png_bytes, provenance, timing = render(result["final_prompt"], spec.seed)
    out_path = out_dir / f"{spec.id}.png"
    out_path.write_bytes(png_bytes)
    print(f"saved {out_path} ({len(png_bytes)} bytes)")
    print(f"provenance: {json.dumps(provenance)}")
    print(f"timing: {json.dumps(timing)}")

    result["provenance"] = provenance
    result["timing"] = timing
    (out_dir / f"{spec.id}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
