"""
training/vlm_dataset_builder.py — Instruction-tuned VLM fine-tuning dataset
==========================================================================
Builds a dataset for fine-tuning Cosmos Reason 2 into an agriculture-specific
vision-language model.  Pipeline:

  1. Render photorealistic farm scenes via Isaac Sim (fallback: procedural)
  2. For each rendered frame, generate agricultural Q&A pairs based on the
     known scene metadata (diseases, growth stage, weather, crop type)
  3. Output instruction-tuned JSONL suitable for VLM fine-tuning

Dataset format (per line in JSONL):
  {
    "image": "path/to/frame.png",
    "conversations": [
      {"from": "human", "value": "<image>\nWhat crop health issues do you see?"},
      {"from": "gpt",   "value": "I can see gray_leaf_spot affecting ..."}
    ],
    "metadata": { ... scene params ... }
  }

This is the ShareGPT/LLaVA conversation format widely used for VLM fine-tuning
(LLaVA, Qwen-VL, InternVL, etc.).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

# ── Structured Q&A question templates ─────────────────────────────────
# Questions that elicit the structured OBSERVATION/DIAGNOSIS/SEVERITY/
# CONFIDENCE/ACTION output format used by ag_perception.py in production.

STRUCTURED_QUESTIONS = [
    "What crop health issues do you see in this aerial image?",
    "Analyze the visible plant stress patterns in this drone image.",
    "Are there any signs of disease or pest damage in this field?",
    "What growth stage is this field in and are there any problems?",
    "Describe what you see in this aerial farm image.",
    "What treatment do you recommend for the issues visible in this field?",
    "What immediate actions should a farmer take based on this image?",
    "Estimate the crop maturity and health from this drone image.",
]

# ── Treatment knowledge base ──────────────────────────────────────────
DISEASE_TREATMENTS = {
    # ── TIER 1: Ohio Corn Diseases ──────────────────────────────────
    "gray_leaf_spot": {
        "treatment": "Apply strobilurin-based fungicide (e.g., azoxystrobin) at labeled rate",
        "severity": "high",
        "urgency": "within 7 days",
    },
    "northern_corn_leaf_blight": {
        "treatment": "Apply triazole fungicide (e.g., propiconazole); consider resistant hybrids for next season",
        "severity": "high",
        "urgency": "within 5 days",
    },
    "common_rust": {
        "treatment": "Apply fungicide if pustules appear before tasseling; monitor spread rate",
        "severity": "high",
        "urgency": "monitor, treat if spreading",
    },
    # ── TIER 2: Ohio Soybean Diseases ───────────────────────────────
    "sudden_death_syndrome": {
        "treatment": "No in-season chemical fix; improve drainage, use resistant varieties next season, apply seed treatment (fluopyram)",
        "severity": "high",
        "urgency": "plan for next season",
    },
    "frogeye_leaf_spot": {
        "treatment": "Apply strobilurin or triazole fungicide at R3 stage; use resistant varieties next season",
        "severity": "moderate",
        "urgency": "within 7 days if spreading",
    },
    # ── TIER 3: Cross-Crop Stress Classes ───────────────────────────
    "nitrogen_deficiency": {
        "treatment": "Side-dress with UAN or urea at 40-60 lb N/acre; tissue test to confirm",
        "severity": "moderate",
        "urgency": "within 5 days during vegetative growth",
    },
    "water_stress": {
        "treatment": "Irrigate if available; reduce nitrogen applications; avoid additional plant stress from herbicides",
        "severity": "moderate",
        "urgency": "immediate if irrigation available",
    },
    "weed_pressure": {
        "treatment": "Apply post-emergent herbicide appropriate for weed species and crop stage; scout to identify dominant weed species first",
        "severity": "moderate",
        "urgency": "within 5 days before canopy closure",
    },
    "stand_gap": {
        "treatment": "Assess extent of gaps; replant if early enough in season and gaps exceed 10% of field; adjust seeding rate for next season",
        "severity": "low",
        "urgency": "evaluate within 7 days of emergence",
    },
    "ponding": {
        "treatment": "Improve field drainage with tile or surface grading; avoid driving equipment on saturated soil; check tile outlets for blockage",
        "severity": "moderate",
        "urgency": "address drainage before next planting; monitor crop for root rot",
    },
    "healthy": {
        "treatment": "No treatment needed; continue standard monitoring schedule",
        "severity": "none",
        "urgency": "routine monitoring",
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Calibration Mixer — generates anti-hallucination training variants
# ═══════════════════════════════════════════════════════════════════════
# Distribution: 50% confident correct, 20% uncertain honest,
#               20% confident negative, 10% edge cases
# Each base pattern → 15-25 variants with different wording, diseases,
# environmental contexts, altitudes, and growth stages.

SYSTEM_PROMPT = (
    "You are an experienced agronomist analyzing aerial drone imagery of "
    "crop fields in central Ohio. You specialize in corn and soybean "
    "pathology. When shown an image, provide a structured assessment with "
    "OBSERVATION, DIAGNOSIS, SEVERITY, CONFIDENCE, and ACTION. Keep "
    "responses under 150 words."
)

# ── Variant pools ──────────────────────────────────────────────────────

_ALTITUDES = [15, 20, 25, 30, 35, 40]
_GROWTH_STAGES_CORN = ["V4", "V6", "V8", "V10", "V12", "VT", "R1", "R3"]
_GROWTH_STAGES_SOY = ["V3", "V5", "R1", "R2", "R3", "R4", "R5"]
_WEATHER = ["clear", "overcast", "hazy", "post-rain", "morning dew"]
_LIGHTING = ["morning", "midday", "afternoon", "golden_hour", "overcast"]
_HUMIDITY = list(range(55, 96, 5))
_TEMPS = list(range(20, 36))
_FIELD_SIZES_ACRES = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

_CORN_DISEASES = {
    "gray_leaf_spot": {
        "pathogen": "Cercospora zeae-maydis",
        "lesion_desc": [
            "rectangular gray-tan lesions running parallel to leaf veins",
            "elongated gray-brown rectangular spots bounded by veins",
            "narrow gray lesions with sharp edges along leaf veins",
            "parallel-sided gray-tan necrotic areas on mid-canopy leaves",
        ],
        "severity_cues": [
            "lower canopy shows heavier infection than upper canopy",
            "lesions coalescing on lower leaves, upper leaves beginning to show symptoms",
            "scattered lesions on mid-canopy, ear leaf not yet affected",
            "heavy infection throughout canopy, ear leaf showing early symptoms",
        ],
        "treatment": "strobilurin fungicide (e.g., Headline AMP, azoxystrobin)",
        "timeline": "within 3-5 days before infection reaches ear leaf",
    },
    "northern_corn_leaf_blight": {
        "pathogen": "Exserohilum turcicum",
        "lesion_desc": [
            "large cigar-shaped gray-green lesions 2-6 inches long",
            "elongated elliptical gray-green to tan necrotic areas",
            "long spindle-shaped lesions with wavy margins",
            "cigar-shaped lesions starting from lower leaves, progressing upward",
        ],
        "severity_cues": [
            "lesions concentrated on lower leaves, upper canopy still clean",
            "moderate spread across mid-canopy, some lesions reaching flag leaf",
            "heavy lesion load on all leaf layers, canopy thinning from necrosis",
            "early infection with scattered lesions on lower 3-4 leaves",
        ],
        "treatment": "triazole fungicide (e.g., propiconazole); plan resistant hybrids next season",
        "timeline": "within 5 days",
    },
    "common_rust": {
        "pathogen": "Puccinia sorghi",
        "lesion_desc": [
            "small circular reddish-brown pustules on both leaf surfaces",
            "raised red-brown oval pustules scattered across leaves",
            "brick-red to cinnamon-brown powdery pustules in clusters",
            "small elongate rust-colored eruptions on upper and lower leaf surface",
        ],
        "severity_cues": [
            "scattered pustules on mid-canopy leaves, not yet heavy",
            "dense pustule clusters on lower leaves, moderate on upper",
            "early pustule formation on a few leaves, spread rate unclear",
            "heavy rust across canopy giving reddish tint visible from altitude",
        ],
        "treatment": "fungicide application if pustules appear before tasseling; monitor spread rate",
        "timeline": "monitor daily, treat within 3 days if spreading rapidly",
    },
}

_SOY_DISEASES = {
    "sudden_death_syndrome": {
        "pathogen": "Fusarium virguliforme",
        "lesion_desc": [
            "interveinal chlorosis and necrosis with veins remaining green",
            "yellow-brown patches between leaf veins, leaves still attached to stems",
            "rapid interveinal yellowing progressing to brown necrosis",
            "mottled yellow-green pattern with necrotic interveinal tissue",
        ],
        "severity_cues": [
            "scattered symptomatic plants in low-lying area of field",
            "dense patch of affected plants, 0.3-0.5 acres, concentrated in drainage low",
            "early foliar symptoms appearing, root rot likely established",
            "advanced symptoms with significant defoliation in wet zone",
        ],
        "treatment": "no in-season chemical fix; improve drainage, use resistant varieties and fluopyram seed treatment next season",
        "timeline": "plan for next season",
    },
    "frogeye_leaf_spot": {
        "pathogen": "Cercospora sojina",
        "lesion_desc": [
            "circular spots with gray centers and dark reddish-brown borders",
            "small round lesions with light gray centers ringed by dark margins",
            "scattered 'frogeye' spots with distinct dark borders on upper leaves",
            "multiple circular gray-centered spots per leaf, some coalescing",
        ],
        "severity_cues": [
            "light infection on upper canopy, lesions not yet coalescing",
            "moderate spread with multiple lesions per leaf",
            "heavy infection with lesions merging, defoliation beginning",
            "early-stage scattered spots on newer leaves",
        ],
        "treatment": "strobilurin or triazole fungicide at R3; use resistant varieties next season",
        "timeline": "within 7 days if spreading",
    },
}

_STRESS_CONDITIONS = {
    "nitrogen_deficiency": {
        "visual_desc": [
            "V-shaped yellowing from leaf tips on lower leaves",
            "pale yellow-green streaks following fertilizer application rows",
            "lower leaf chlorosis with green upper canopy",
            "firing of lower leaves with yellow-green midrib area",
        ],
        "severity_cues": [
            "yellowing limited to bottom 2-3 leaves, upper canopy green",
            "pronounced chlorosis through lower half of plant, streaking pattern visible",
            "severe deficiency with lower leaves necrotic, midcanopy yellowing",
            "mild V-shaped tip yellowing on lower leaves only",
        ],
        "treatment": "side-dress 40-60 lb N/acre UAN or urea; tissue test to confirm",
        "timeline": "within 5-7 days during vegetative growth",
    },
    "water_stress": {
        "visual_desc": [
            "leaf rolling and gray-green color shift across canopy",
            "wilted plants with dull gray-green appearance, canopy thinning",
            "corn leaves curled inward, reduced canopy reflectance",
            "gray-green shift in elevated areas of field that drain faster",
        ],
        "severity_cues": [
            "mild leaf rolling in afternoon, recovery expected overnight",
            "persistent leaf rolling, canopy visibly thinner than adjacent rows",
            "severe wilting with gray-green discoloration, no overnight recovery",
            "localized stress in higher elevation zones, rest of field OK",
        ],
        "treatment": "irrigate if available; reduce nitrogen applications; avoid herbicide stress",
        "timeline": "immediate if irrigation available",
    },
}

# ── Why uncertain + what resolves it ──────────────────────────────────

_UNCERTAINTY_REASONS = [
    {
        "why": "altitude is too high to resolve individual lesion morphology",
        "resolve": "Dispatch a spot-check flight at 15m altitude over this zone for higher-resolution imagery",
    },
    {
        "why": "lesion shape is ambiguous between multiple pathogens at this resolution",
        "resolve": "Scout this zone on foot within 48 hours and photograph lesions at leaf level for pathogen identification",
    },
    {
        "why": "lighting conditions create shadows that obscure canopy color",
        "resolve": "Re-fly this zone during midday under clear skies for accurate color assessment",
    },
    {
        "why": "discoloration could be nutrient stress, early disease, or natural hybrid variation",
        "resolve": "Pull tissue samples from 5-10 affected and adjacent healthy plants and submit to the OSU Plant Diagnostic Lab",
    },
    {
        "why": "environmental conditions favor disease but no visible symptoms are confirmed yet",
        "resolve": "Increase scouting frequency to every 3-4 days while humidity remains above 85%",
    },
    {
        "why": "detector confidence is below 50% and the visual signal is faint",
        "resolve": "Mark this location for ground scouting within the next week; do NOT treat based on this detection alone",
    },
    {
        "why": "the affected area is at the edge of the flight path where image quality degrades",
        "resolve": "Re-fly with this zone centered in the flight path for full-resolution coverage",
    },
    {
        "why": "post-rain moisture on leaves is altering canopy reflectance and color",
        "resolve": "Wait 24-48 hours for foliage to dry and re-fly; wet leaves confound color-based diagnosis",
    },
    {
        "why": "growth stage transition makes it difficult to distinguish stress from normal maturation changes",
        "resolve": "Scout on foot to inspect individual plants and determine if discoloration is senescence or pathology",
    },
    {
        "why": "multiple stress factors may be overlapping in this zone (nutrient + disease + water)",
        "resolve": "Pull tissue samples, check soil moisture at 6-inch depth, and photograph individual symptomatic leaves for differential diagnosis",
    },
    {
        "why": "the canopy density is too high to see lower-leaf symptoms where disease typically starts",
        "resolve": "Scout on foot and manually inspect lower canopy leaves; aerial view only shows the top canopy layer",
    },
    {
        "why": "cloud shadow is moving across the detection zone, creating transient dark patches",
        "resolve": "Re-fly this zone in stable lighting conditions or wait for cloud cover to clear",
    },
]

# ── What triggered the false positive ─────────────────────────────────

_FALSE_TRIGGERS = [
    {
        "trigger": "tree line shadow",
        "explanation": "A shadow cast by the tree line along the field edge created a darker band across several rows, mimicking the contrast pattern of water stress",
    },
    {
        "trigger": "hybrid boundary",
        "explanation": "Two different corn hybrids planted adjacent to each other show slightly different green tones from the air — the linear transition at the planting boundary triggered the nutrient deficiency detector",
    },
    {
        "trigger": "equipment track",
        "explanation": "Wheel tracks from a recent spray pass compressed the canopy, creating parallel lines of slightly shorter plants that the detector read as stand gaps",
    },
    {
        "trigger": "center pivot track",
        "explanation": "The center pivot wheel track creates a narrow strip of compacted soil and shorter plants in a gentle arc through the field, triggering the stand gap detector",
    },
    {
        "trigger": "morning dew reflection",
        "explanation": "Dew on leaf surfaces in early morning created reflective highlights that altered the apparent canopy color, triggering a false nutrient deficiency detection",
    },
    {
        "trigger": "camera lens flare",
        "explanation": "Low-angle sun caused lens flare in the corner of the frame, creating a warm-toned artifact that the detector interpreted as rust discoloration",
    },
    {
        "trigger": "harvest residue",
        "explanation": "Last season's crop residue visible between rows in early-season imagery created brown patches that triggered the stand gap detector",
    },
    {
        "trigger": "waterway grass strip",
        "explanation": "A grass waterway running through the field shows different vegetation texture and color than the crop rows, triggering the weed pressure detector",
    },
    {
        "trigger": "soil color variation",
        "explanation": "Natural variation in soil type across the field creates darker and lighter bands between rows that mimic nutrient deficiency patterns from the air",
    },
    {
        "trigger": "planter skip",
        "explanation": "A single-row planter skip from a plugged seed tube was flagged as disease stress, but the adjacent rows are fully healthy — this is a mechanical issue, not pathology",
    },
    {
        "trigger": "irrigation riser shadow",
        "explanation": "Shadows cast by irrigation system risers created small dark spots at regular intervals that triggered the disease detector",
    },
    {
        "trigger": "bird flock on field",
        "explanation": "A flock of birds resting on the canopy created dark spots that the detector flagged as disease patches",
    },
]

# ── Edge case scenarios ───────────────────────────────────────────────

_EDGE_CASES = [
    {
        "scenario": "multiple diseases co-occurring",
        "desc": "Both gray leaf spot and northern corn leaf blight present simultaneously on the same plant population",
        "observation": "Mixed lesion morphology visible — some rectangular vein-bounded lesions alongside larger cigar-shaped lesions on the same plants. Both pathogens appear active.",
        "response_note": "Multiple pathogens confirmed. Both GLS and NCLB present, which complicates resistance gene selection for next season but simplifies immediate treatment — broad-spectrum strobilurin covers both.",
        "confidence": "Medium-High",
        "action": "Apply broad-spectrum strobilurin fungicide within 3 days. Scout on foot to confirm both pathogens and document resistance breakdown for hybrid selection next season.",
    },
    {
        "scenario": "disease at field border only",
        "desc": "Disease symptoms visible only in the outer 4-6 rows along field edges, interior clean",
        "observation": "Foliar disease symptoms confined to field perimeter rows. Interior canopy appears uniformly healthy. Edge effect pattern is consistent with spore source from adjacent field or grass waterway.",
        "response_note": "Border-only infection. Disease is present but confined to edge rows where spore pressure from windbreaks or adjacent infected fields is highest. Interior may stay clean if conditions improve.",
        "confidence": "High",
        "action": "Spot-spray perimeter rows only — do not treat entire field. Monitor interior weekly. This saves 80-90% of fungicide cost compared to blanket application.",
    },
    {
        "scenario": "early season bare soil with no canopy",
        "desc": "Very early growth stage (V2-V3) with minimal canopy, mostly bare soil visible",
        "observation": "Crop is in early emergence (V2-V3). Canopy coverage is less than 15% with bare soil dominating the aerial view. Individual plants are small and widely spaced.",
        "response_note": "Too early for meaningful disease assessment from aerial imagery. Plant canopy is insufficient for foliar disease detection. Focus assessment on emergence uniformity and stand count.",
        "confidence": "High that no foliar disease assessment is possible at this stage",
        "action": "Switch to stand count mode — estimate plants per acre from the aerial view. Flag any gaps exceeding 2 feet in the row for replant evaluation. Foliar disease scouting should begin at V6 or later.",
    },
    {
        "scenario": "post-hail damage mimicking disease",
        "desc": "Hail damage creating shredded leaves and bruised tissue that resembles disease lesions",
        "observation": "Irregular tattered leaf edges and white-to-brown bruised patches scattered across the canopy. Damage pattern is uniform across the entire visible area rather than clustered.",
        "response_note": "Physical damage from hail, not disease. The uniform distribution across the field and the torn/shredded leaf edges are diagnostic — disease spreads in gradients from inoculum sources, not uniformly.",
        "confidence": "High. Pattern is consistent with hail, not pathology.",
        "action": "Document damage extent for crop insurance claim. Monitor closely over next 14 days — hail wounds are entry points for fungal pathogens. Preventive fungicide application may be warranted if humidity remains high.",
    },
    {
        "scenario": "herbicide drift damage",
        "desc": "Herbicide drift from adjacent field causing leaf curl and bleaching on field border",
        "observation": "Leaf curling, cupping, and bleaching visible in a gradient pattern along one edge of the field. Symptom severity decreases with distance from the field boundary. Affected area is approximately 4-8 rows deep.",
        "response_note": "Herbicide drift injury, not disease or nutrient stress. The gradient pattern from the field edge inward is diagnostic — this is an off-target application from the adjacent field.",
        "confidence": "High. Gradient pattern from boundary is diagnostic of drift.",
        "action": "Document damage with geo-tagged imagery for potential drift complaint. Monitor affected rows — most crops recover from mild drift within 10-14 days. Do NOT apply additional herbicide to the affected area.",
    },
    {
        "scenario": "late-season senescence confused with disease",
        "desc": "Normal late-season yellowing and dry-down mistaken for disease",
        "observation": "Canopy showing yellow-brown coloration across a broad area. Plants appear to be in late reproductive stage (R5-R6). Lower leaves are senescing naturally.",
        "response_note": "Normal maturation senescence, not disease. At this growth stage, lower leaf yellowing and brown-down is expected. The uniform progression from bottom to top and the late calendar date confirm natural dry-down.",
        "confidence": "High. Growth stage and calendar date confirm natural senescence.",
        "action": "No treatment needed. Begin monitoring for harvest readiness. Check grain moisture in 7-10 days. Natural senescence should not be confused with disease — no fungicide application is warranted at this stage.",
    },
    {
        "scenario": "sensor malfunction producing artifacts",
        "desc": "Camera sensor producing color banding or hot pixels that could be misread",
        "observation": "Regular geometric pattern of discoloration visible — horizontal banding or checkerboard artifacts that do not follow field geometry. Pattern is grid-aligned, not following rows or terrain.",
        "response_note": "Sensor artifact, not field condition. The geometric regularity of the pattern is impossible in nature — crop stress follows soil, drainage, and wind patterns, never perfect grid lines.",
        "confidence": "High that this is a sensor/processing artifact.",
        "action": "Discard this frame and recalibrate the camera sensor. Check for overheating or firmware issues. Re-fly this zone with a functioning sensor before making any field assessments.",
    },
]

# Question variants for calibration conversations
_CAL_QUESTIONS = [
    "Analyze this aerial image of a {crop} field. Flight altitude: {alt}m. "
    "Environmental: {temp}°C, {humidity}% humidity, VPD {vpd:.2f} kPa. "
    "Fungal disease risk level: {risk}. {detection_note} Provide your assessment.",

    "Review this drone capture of a {crop} field taken at {alt}m. "
    "Conditions: {temp}°C, {humidity}% RH, VPD {vpd:.2f} kPa. "
    "Disease pressure: {risk}. {detection_note} What do you see?",

    "Assess this {alt}m altitude image of {crop}. "
    "Environment: {temp}°C, {humidity}% humidity, VPD {vpd:.2f} kPa. "
    "Risk level: {risk}. {detection_note} Give your structured analysis.",
]


def _calc_vpd(temp_c: float, humidity_pct: float) -> float:
    """Approximate vapor pressure deficit in kPa."""
    svp = 0.6108 * (2.7183 ** ((17.27 * temp_c) / (temp_c + 237.3)))
    return svp * (1 - humidity_pct / 100)


def _fungal_risk(humidity: int, temp: int) -> str:
    if humidity >= 85 and 20 <= temp <= 30:
        return "high"
    elif humidity >= 70:
        return "moderate"
    return "low"


class CalibrationMixer:
    """Generate calibration training variants and mix into VLM dataset.

    Produces conversations at the 50/20/20/10 distribution:
      - 50% confident correct diagnoses
      - 20% uncertain honest (with specific WHY + resolution)
      - 20% confident negatives (with specific false trigger explanation)
      - 10% edge cases
    """

    def __init__(self, calibration_yaml_path: Path,
                 variants_per_pattern: tuple[int, int] = (15, 25)):
        self.variants_range = variants_per_pattern
        self.base_patterns = []
        if calibration_yaml_path.exists():
            with open(calibration_yaml_path) as f:
                self.base_patterns = yaml.safe_load(f) or []

    def _random_env(self) -> dict:
        """Generate a randomized environmental context."""
        temp = random.choice(_TEMPS)
        humidity = random.choice(_HUMIDITY)
        alt = random.choice(_ALTITUDES)
        weather = random.choice(_WEATHER)
        lighting = random.choice(_LIGHTING)
        vpd = _calc_vpd(temp, humidity)
        risk = _fungal_risk(humidity, temp)
        return dict(temp=temp, humidity=humidity, alt=alt, weather=weather,
                    lighting=lighting, vpd=vpd, risk=risk)

    def _random_question(self, crop: str, env: dict,
                         detection_note: str) -> str:
        template = random.choice(_CAL_QUESTIONS)
        return template.format(crop=crop, detection_note=detection_note, **env)

    # ── Confident correct variants ────────────────────────────────────

    def _gen_confident_correct(self, n: int) -> list[dict]:
        """Generate n confident-correct calibration conversations."""
        records = []
        all_diseases = {**_CORN_DISEASES, **_SOY_DISEASES, **_STRESS_CONDITIONS}

        for _ in range(n):
            disease_key = random.choice(list(all_diseases))
            disease = all_diseases[disease_key]
            crop = "corn" if disease_key in _CORN_DISEASES else \
                   "soybeans" if disease_key in _SOY_DISEASES else \
                   random.choice(["corn", "soybeans"])
            env = self._random_env()
            stages = _GROWTH_STAGES_CORN if crop == "corn" else _GROWTH_STAGES_SOY
            stage = random.choice(stages)
            field_acres = random.choice(_FIELD_SIZES_ACRES)
            lesion_desc = random.choice(disease.get("lesion_desc",
                                        disease.get("visual_desc")))
            severity_cue = random.choice(disease.get("severity_cues",
                                         ["moderate spread across canopy"]))
            det_conf = random.randint(65, 92)
            det_note = (f"The object detector flagged: "
                        f"{disease_key.replace('_', ' ')} at {det_conf}% confidence.")

            sev_num = random.choice([3, 4, 5, 6, 7, 8])
            sev_word = "Mild" if sev_num <= 3 else "Moderate" if sev_num <= 6 else "Severe"

            observation = (
                f"{lesion_desc.capitalize()} visible on {crop} at "
                f"{env['alt']}m altitude under {env['weather']} conditions. "
                f"Crop is at {stage} stage. {severity_cue.capitalize()}. "
                f"Affected area covers approximately {field_acres} acres."
            )

            pathogen = disease.get("pathogen", "")
            pathogen_note = f" ({pathogen})" if pathogen else ""
            diag_name = disease_key.replace("_", " ").title()

            diagnosis = (
                f"{diag_name}{pathogen_note}. "
                f"{lesion_desc.capitalize()} — this presentation is "
                f"characteristic. Environmental conditions "
                f"({'favor' if env['risk'] in ('high', 'moderate') else 'do not strongly favor'}) "
                f"continued development."
            )

            confidence_val = round(random.uniform(0.78, 0.95), 2)
            treatment = disease.get("treatment", "Consult agronomist")
            timeline = disease.get("timeline", "within 7 days")

            action = (
                f"Scout this zone on foot to confirm within 24 hours. "
                f"If confirmed, {treatment} ({timeline}). "
                f"Spot-treat this {field_acres}-acre zone only — "
                f"do not spray the entire field."
            )

            answer = (
                f"OBSERVATION: {observation}\n\n"
                f"DIAGNOSIS: {diagnosis}\n\n"
                f"SEVERITY: {sev_num}/10. {sev_word} — {severity_cue}.\n\n"
                f"CONFIDENCE: High ({confidence_val}). Lesion morphology is "
                f"distinctive and consistent with known presentation.\n\n"
                f"ACTION: {action}"
            )

            question = self._random_question(crop, env, det_note)

            records.append({
                "image": None,  # assigned during mixing
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "calibration_category": "confident_correct",
                    "disease": disease_key,
                    "crop_type": crop,
                    "altitude_m": env["alt"],
                    "growth_stage": stage,
                },
            })

        return records

    # ── Uncertain honest variants ─────────────────────────────────────

    def _gen_uncertain_honest(self, n: int) -> list[dict]:
        """Generate n uncertain-honest calibration conversations.

        Every example states specifically WHY it's uncertain and WHAT
        would resolve the uncertainty.
        """
        records = []
        all_diseases = {**_CORN_DISEASES, **_SOY_DISEASES, **_STRESS_CONDITIONS}

        for _ in range(n):
            disease_key = random.choice(list(all_diseases))
            disease = all_diseases[disease_key]
            crop = "corn" if disease_key in _CORN_DISEASES else \
                   "soybeans" if disease_key in _SOY_DISEASES else \
                   random.choice(["corn", "soybeans"])
            env = self._random_env()
            stages = _GROWTH_STAGES_CORN if crop == "corn" else _GROWTH_STAGES_SOY
            stage = random.choice(stages)

            uncertainty = random.choice(_UNCERTAINTY_REASONS)
            det_conf = random.randint(35, 55)
            det_note = (f"The object detector flagged: "
                        f"{disease_key.replace('_', ' ')} at {det_conf}% "
                        f"confidence.")

            # Pick a plausible alternative diagnosis
            alt_keys = [k for k in all_diseases if k != disease_key]
            alt_disease = random.choice(alt_keys).replace("_", " ").title()
            primary_name = disease_key.replace("_", " ").title()

            observation = (
                f"Subtle color variation visible in {crop} canopy at "
                f"{env['alt']}m altitude under {env['weather']} conditions. "
                f"Crop is at {stage} stage. A patch approximately "
                f"{random.choice([10, 15, 20, 25, 30])} meters across shows "
                f"faint discoloration. No clear lesion patterns are "
                f"distinguishable at this altitude and resolution."
            )

            diagnosis = (
                f"Inconclusive. This could be early-stage {primary_name}, "
                f"{alt_disease.lower()}, natural maturation variation, or an "
                f"imaging artifact. Specific reason for uncertainty: "
                f"{uncertainty['why']}. A detector confidence of {det_conf}% "
                f"reflects this ambiguity."
            )

            confidence_val = round(random.uniform(0.25, 0.50), 2)

            action = (
                f"To resolve this uncertainty: {uncertainty['resolve']}. "
                f"Do NOT apply any treatment based on this detection alone — "
                f"the evidence does not justify the cost. If ground scouting "
                f"confirms disease, escalate to treatment protocol at that point."
            )

            answer = (
                f"OBSERVATION: {observation}\n\n"
                f"DIAGNOSIS: {diagnosis}\n\n"
                f"SEVERITY: Unknown without ground confirmation.\n\n"
                f"CONFIDENCE: Low ({confidence_val}). {uncertainty['why'].capitalize()}.\n\n"
                f"ACTION: {action}"
            )

            question = self._random_question(crop, env, det_note)

            records.append({
                "image": None,
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "calibration_category": "uncertain_honest",
                    "uncertainty_reason": uncertainty["why"],
                    "resolution_action": uncertainty["resolve"],
                    "disease_candidate": disease_key,
                    "crop_type": crop,
                    "altitude_m": env["alt"],
                    "growth_stage": stage,
                },
            })

        return records

    # ── Confident negative variants ───────────────────────────────────

    def _gen_confident_negative(self, n: int) -> list[dict]:
        """Generate n confident-negative calibration conversations.

        Every example explains specifically WHAT triggered the false positive.
        """
        records = []
        all_diseases = {**_CORN_DISEASES, **_SOY_DISEASES, **_STRESS_CONDITIONS}

        for _ in range(n):
            false_trigger = random.choice(_FALSE_TRIGGERS)
            triggered_disease = random.choice(list(all_diseases))
            crop = random.choice(["corn", "soybeans"])
            env = self._random_env()
            stages = _GROWTH_STAGES_CORN if crop == "corn" else _GROWTH_STAGES_SOY
            stage = random.choice(stages)

            det_conf = random.randint(38, 58)
            det_note = (f"The object detector flagged: "
                        f"{triggered_disease.replace('_', ' ')} at {det_conf}% "
                        f"confidence.")

            observation = (
                f"Inspecting the area flagged by the detector in {crop} field "
                f"at {env['alt']}m altitude under {env['weather']} conditions. "
                f"Crop is at {stage} stage. The flagged region shows "
                f"{false_trigger['trigger']} — {false_trigger['explanation']}. "
                f"The surrounding crop appears uniformly healthy with no "
                f"disease symptoms, stress indicators, or canopy abnormalities."
            )

            triggered_name = triggered_disease.replace("_", " ").title()
            diagnosis = (
                f"False positive. The detector triggered on "
                f"{false_trigger['trigger']}, not on actual {triggered_name.lower()}. "
                f"The crop in this zone is healthy. "
                f"The {false_trigger['trigger']} created a visual contrast "
                f"pattern that the model mistook for stress symptoms."
            )

            action = (
                f"No treatment needed. This detection should be logged as a "
                f"confirmed false positive caused by {false_trigger['trigger']}. "
                f"Consider adding this type of artifact to the model's negative "
                f"training set to reduce future false alarms in similar conditions."
            )

            vpd_status = ("in the optimal range" if 0.8 <= env["vpd"] <= 1.5
                          else "elevated" if env["vpd"] > 1.5
                          else "below optimal")

            answer = (
                f"OBSERVATION: {observation}\n\n"
                f"DIAGNOSIS: {diagnosis}\n\n"
                f"SEVERITY: 0/10. No actual stress or disease detected.\n\n"
                f"CONFIDENCE: High that this is a false positive. The flagged "
                f"area corresponds to {false_trigger['trigger']}, and no "
                f"stress symptoms are visible in the surrounding crop.\n\n"
                f"ACTION: {action} Note: VPD is {vpd_status} at "
                f"{env['vpd']:.2f} kPa — "
                f"{'monitor for genuine stress' if env['vpd'] > 1.5 else 'conditions are not concerning'}."
            )

            question = self._random_question(crop, env, det_note)

            records.append({
                "image": None,
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "calibration_category": "confident_negative",
                    "false_trigger": false_trigger["trigger"],
                    "false_trigger_detail": false_trigger["explanation"],
                    "triggered_disease": triggered_disease,
                    "crop_type": crop,
                    "altitude_m": env["alt"],
                    "growth_stage": stage,
                },
            })

        return records

    # ── Edge case variants ────────────────────────────────────────────

    def _gen_edge_cases(self, n: int) -> list[dict]:
        """Generate n edge-case calibration conversations."""
        records = []

        for _ in range(n):
            case = random.choice(_EDGE_CASES)
            crop = random.choice(["corn", "soybeans"])
            env = self._random_env()
            stages = _GROWTH_STAGES_CORN if crop == "corn" else _GROWTH_STAGES_SOY
            stage = random.choice(stages)

            det_note = f"Scenario: {case['scenario']}. {case['desc']}."

            answer = (
                f"OBSERVATION: {case['observation']}\n\n"
                f"DIAGNOSIS: {case['response_note']}\n\n"
                f"SEVERITY: Context-dependent — see diagnosis.\n\n"
                f"CONFIDENCE: {case['confidence']}.\n\n"
                f"ACTION: {case['action']}"
            )

            question = self._random_question(crop, env, det_note)

            records.append({
                "image": None,
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "metadata": {
                    "calibration_category": "edge_case",
                    "scenario": case["scenario"],
                    "crop_type": crop,
                    "altitude_m": env["alt"],
                    "growth_stage": stage,
                },
            })

        return records

    # ── Main mixing entry point ───────────────────────────────────────

    def generate_calibration_set(self, total_target: int) -> list[dict]:
        """Generate calibration conversations at the 50/20/20/10 distribution.

        Args:
            total_target: Total number of calibration conversations to produce.

        Returns:
            List of ShareGPT conversation records ready for JSONL output.
        """
        n_confident = int(total_target * 0.50)
        n_uncertain = int(total_target * 0.20)
        n_negative = int(total_target * 0.20)
        n_edge = total_target - n_confident - n_uncertain - n_negative

        print(f"[Calibration] Generating {total_target} calibration conversations:")
        print(f"  Confident correct: {n_confident} (50%)")
        print(f"  Uncertain honest:  {n_uncertain} (20%)")
        print(f"  Confident negative:{n_negative} (20%)")
        print(f"  Edge cases:        {n_edge} (10%)")

        records = []
        records.extend(self._gen_confident_correct(n_confident))
        records.extend(self._gen_uncertain_honest(n_uncertain))
        records.extend(self._gen_confident_negative(n_negative))
        records.extend(self._gen_edge_cases(n_edge))

        random.shuffle(records)

        print(f"[Calibration] Generated {len(records)} calibration conversations")
        return records


# ── Growth stage heuristics ───────────────────────────────────────────
def _classify_growth_stage(plant_height: float, plant_density: float,
                           crop_type: str) -> dict:
    """Derive growth stage description from scene parameters."""
    if plant_height < 0.5:
        stage = "early vegetative (V2-V4)"
        row_vis = "clearly visible with bare soil between rows"
        color = "light green with visible soil"
        pct = 0.15
    elif plant_height < 1.0:
        stage = "mid-vegetative (V6-V10)"
        row_vis = "visible but canopy beginning to close"
        color = "medium green with increasing canopy coverage"
        pct = 0.4
    elif plant_height < 1.5:
        stage = "late vegetative to early reproductive"
        row_vis = "partially obscured by canopy closure"
        color = "deep green indicating active photosynthesis"
        pct = 0.65
    else:
        stage = "reproductive (R1-R3)" if crop_type == "soybean" else "tasseling/silking (VT-R1)"
        row_vis = "mostly hidden under dense canopy"
        color = "deep green, uniformly dense canopy"
        pct = 0.85

    return {
        "growth_stage": stage,
        "row_visibility": row_vis,
        "color_assessment": color,
        "growth_pct": pct,
    }


def _lighting_time_description(lighting: str) -> str:
    """Map lighting preset name to natural language."""
    return {
        "morning": "early morning with low-angle eastern sun",
        "midday": "midday with overhead sun and minimal shadows",
        "afternoon": "afternoon with western sun creating angled shadows",
        "golden_hour": "golden hour with warm, low-angle light",
        "overcast": "overcast sky providing diffuse, even illumination",
    }.get(lighting, lighting)


class VLMDatasetBuilder:
    """Build instruction-tuned fine-tuning dataset for agricultural VLM."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vlm_cfg = cfg.get("vlm_finetune", {})
        self.synth_cfg = cfg.get("synthetic", {})
        self.disease_classes = self.synth_cfg.get("disease_classes", [])

        data_root = Path(cfg["paths"]["data_root"]).expanduser()
        self.output_dir = data_root / "vlm_finetune"
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.qa_per_image = self.vlm_cfg.get("qa_pairs_per_image", 3)
        self.include_general = self.vlm_cfg.get("include_general_description", True)

        # VLM domain-matched output: resize to match ROS2→Cosmos pipeline
        self.vlm_resolution = self.vlm_cfg.get("vlm_resolution", 256)
        self.vlm_jpeg_quality = self.vlm_cfg.get("vlm_jpeg_quality", 70)

    def _build_structured_response(self, scene: dict) -> str:
        """Build a structured OBSERVATION/DIAGNOSIS/SEVERITY/CONFIDENCE/ACTION
        response from scene metadata, matching the format parsed by
        ag_perception.py in production.

        Each field is generated dynamically from the scene's disease patches,
        growth stage, and the DISEASE_TREATMENTS knowledge base.
        """
        diseases = scene.get("diseases", [])
        field = scene.get("field", {})
        camera = scene.get("camera", {})

        crop_type = scene.get("crop_type", "corn")
        altitude_m = camera.get("z", scene.get("altitude_m", 50))
        weather = scene.get("weather", "clear")
        lighting = scene.get("lighting", "midday")
        plant_height_m = field.get("plant_height_m", 1.0)
        plant_density = field.get("plant_density", 0.8)

        growth = _classify_growth_stage(plant_height_m, plant_density, crop_type)
        time_desc = _lighting_time_description(lighting)
        has_issues = any(d["disease"] != "healthy" for d in diseases)

        # ── OBSERVATION ──────────────────────────────────────────────
        disease_names = [d["disease"].replace("_", " ") for d in diseases]
        if has_issues:
            obs_issues = ", ".join(
                f"{d['disease'].replace('_', ' ')} ({d.get('num_instances', 1)} "
                f"zone(s) spanning ~{d.get('radius_m', 5):.0f}m radius)"
                for d in diseases if d["disease"] != "healthy"
            )
            observation = (
                f"Aerial view of {crop_type} field at {altitude_m:.0f}m altitude "
                f"under {weather} conditions with {time_desc}. "
                f"Crop is in {growth['growth_stage']} stage with "
                f"{plant_density:.0%} canopy density and {plant_height_m:.1f}m "
                f"plant height. Visible stress patterns: {obs_issues}."
            )
        else:
            observation = (
                f"Aerial view of {crop_type} field at {altitude_m:.0f}m altitude "
                f"under {weather} conditions with {time_desc}. "
                f"Crop is in {growth['growth_stage']} stage with "
                f"{plant_density:.0%} canopy density and {plant_height_m:.1f}m "
                f"plant height. Canopy appears uniform and healthy with no "
                f"visible stress patterns."
            )

        # ── DIAGNOSIS ────────────────────────────────────────────────
        if has_issues:
            diag_parts = []
            for d in diseases:
                if d["disease"] == "healthy":
                    continue
                name = d["disease"].replace("_", " ").title()
                info = DISEASE_TREATMENTS.get(d["disease"], {})
                diag_parts.append(f"{name} (severity: {info.get('severity', 'moderate')})")
            diagnosis = "; ".join(diag_parts)
        else:
            diagnosis = "No disease or stress detected. Field appears healthy."

        # ── SEVERITY ─────────────────────────────────────────────────
        severities = [DISEASE_TREATMENTS.get(d["disease"], {}).get("severity", "none")
                      for d in diseases]
        if "high" in severities:
            severity = "high"
        elif "moderate" in severities:
            severity = "moderate"
        elif any(s != "none" for s in severities):
            severity = "low"
        else:
            severity = "low"

        # ── CONFIDENCE ───────────────────────────────────────────────
        # Higher altitude → lower confidence; good lighting → higher
        base_conf = 0.85
        if altitude_m > 60:
            base_conf -= 0.10
        elif altitude_m < 35:
            base_conf += 0.05
        if lighting in ("midday", "afternoon"):
            base_conf += 0.05
        elif lighting in ("golden_hour", "overcast"):
            base_conf -= 0.05
        if weather in ("light_rain", "hazy"):
            base_conf -= 0.10
        # More disease instances → clearer signal
        total_instances = sum(d.get("num_instances", 1) for d in diseases
                              if d["disease"] != "healthy")
        if total_instances > 10:
            base_conf += 0.05
        confidence = max(0.3, min(0.98, base_conf + random.uniform(-0.05, 0.05)))

        # ── ACTION ───────────────────────────────────────────────────
        if has_issues:
            action_parts = []
            for d in diseases:
                if d["disease"] == "healthy":
                    continue
                info = DISEASE_TREATMENTS.get(d["disease"], {})
                treatment = info.get("treatment", "Consult agronomist for diagnosis")
                urgency = info.get("urgency", "as soon as practical")
                name = d["disease"].replace("_", " ").title()
                action_parts.append(f"{name}: {treatment} (Timeline: {urgency})")
            action = ". ".join(action_parts) + "."
        else:
            action = ("No treatment needed. Continue standard monitoring schedule. "
                      "Next scouting pass recommended in 5-7 days.")

        return (
            f"OBSERVATION: {observation}\n"
            f"DIAGNOSIS: {diagnosis}\n"
            f"SEVERITY: {severity}\n"
            f"CONFIDENCE: {confidence:.2f}\n"
            f"ACTION: {action}"
        )

    def _build_qa_pairs(self, scene: dict) -> list[dict]:
        """Generate Q&A conversation pairs for one scene.

        Each pair uses a different question but the same structured response
        format (OBSERVATION/DIAGNOSIS/SEVERITY/CONFIDENCE/ACTION) so that
        the fine-tuned VLM learns to always produce structured output
        regardless of how the question is phrased.
        """
        # Select random questions up to qa_per_image
        questions = random.sample(
            STRUCTURED_QUESTIONS,
            min(self.qa_per_image, len(STRUCTURED_QUESTIONS)),
        )

        pairs = []
        for question in questions:
            answer = self._build_structured_response(scene)
            pairs.append({"question": question, "answer": answer})

        return pairs

    def _scene_to_conversations(self, scene: dict,
                                 image_filename: str) -> list[dict]:
        """Convert scene + Q&A into ShareGPT conversation format entries.

        Returns one JSONL record per Q&A pair (multi-turn within each).
        """
        qa_pairs = self._build_qa_pairs(scene)

        records = []
        for qa in qa_pairs:
            record = {
                "image": image_filename,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{qa['question']}"},
                    {"from": "gpt", "value": qa["answer"]},
                ],
                "metadata": {
                    "crop_type": scene.get("crop_type"),
                    "diseases": [d["disease"] for d in scene.get("diseases", [])],
                    "lighting": scene.get("lighting"),
                    "weather": scene.get("weather"),
                    "altitude_m": scene.get("camera", {}).get("z",
                                    scene.get("altitude_m")),
                },
            }
            records.append(record)

        return records

    def build(self, num_images: int = 100, headless: bool = True) -> None:
        """Generate images + instruction-tuned Q&A dataset.

        Args:
            num_images: Number of scenes to render.
            headless: Run Isaac Sim without GUI.
        """
        print(f"[VLM-Dataset] Building instruction-tuned dataset ({num_images} images)")
        print(f"[VLM-Dataset] Q&A pairs per image: {self.qa_per_image}")
        print(f"[VLM-Dataset] Output: {self.output_dir}")

        # ── Step 1: Render scenes (Isaac Sim preferred, procedural fallback)
        try:
            from simulation.isaac_scene_generator import IsaacSceneGenerator
            isaac_gen = IsaacSceneGenerator(self.cfg)
            isaac_available = isaac_gen._check_isaac_available()
        except Exception as e:
            print(f"[VLM-Dataset] Isaac Sim check skipped: {e}")
            isaac_available = False

        # Generate scene metadata (we need it for Q&A regardless of renderer)
        scenes = [isaac_gen._random_scene_params(i) for i in range(num_images)] \
            if isaac_available or 'isaac_gen' in dir() else []

        if isaac_available:
            print("[VLM-Dataset] Rendering via Isaac Sim (photorealistic)")
            # Redirect Isaac output to our VLM images dir
            isaac_gen.output_dir = self.output_dir
            for sub in ["images", "labels", "depth", "segmentation", "metadata"]:
                (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
            isaac_gen.generate(num_images=num_images, headless=headless)
            # Post-process Isaac PNGs: resize + convert to JPEG for domain match
            vlm_size = (self.vlm_resolution, self.vlm_resolution)
            for png_path in self.images_dir.glob("*.png"):
                img = Image.open(png_path)
                img = img.resize(vlm_size, Image.LANCZOS)
                jpg_path = png_path.with_suffix(".jpg")
                img.save(jpg_path, "JPEG", quality=self.vlm_jpeg_quality)
                png_path.unlink()
        else:
            print("[VLM-Dataset] Isaac Sim unavailable — using procedural rendering")
            from data.synthetic_generator import SyntheticGenerator
            gen = SyntheticGenerator(self.cfg)

            # We need scene params for Q&A even in procedural mode
            from simulation.isaac_scene_generator import IsaacSceneGenerator
            isaac_gen = IsaacSceneGenerator(self.cfg)
            scenes = [isaac_gen._random_scene_params(i) for i in range(num_images)]

            for i, scene in enumerate(scenes):
                img = gen._generate_base_field_fast(scene={
                    "growth_stage": scene["field"]["plant_density"],
                    "sun_angle": scene["lighting_params"].get("sun_elevation", 45),
                    "color_temp": "warm" if scene["lighting"] == "golden_hour" else "neutral",
                })
                for patch in scene["diseases"]:
                    gen._overlay_anomaly(img, patch["disease"])
                img = gen._apply_augmentation(img)

                # Resize to VLM resolution and save as JPEG to match
                # production ROS2→Cosmos pipeline (domain gap mitigation)
                vlm_size = (self.vlm_resolution, self.vlm_resolution)
                img = img.resize(vlm_size, Image.LANCZOS)
                img_name = f"vlm_{i:05d}.jpg"
                img.save(self.images_dir / img_name, "JPEG",
                         quality=self.vlm_jpeg_quality)

                if (i + 1) % 50 == 0:
                    print(f"  [{i + 1}/{num_images}] rendered")

        # ── Step 2: Load scene metadata ──────────────────────────────
        # If Isaac rendered, load metadata from its output; otherwise use
        # the scenes list we already have.
        if isaac_available:
            meta_path = self.output_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    all_meta = json.load(f)
                scenes = [m.get("scene_params", m) for m in all_meta]

        # ── Step 3: Build Q&A dataset ────────────────────────────────
        print("[VLM-Dataset] Generating instruction-tuned Q&A pairs...")

        all_records = []
        for i, scene in enumerate(scenes):
            if isaac_available:
                img_filename = f"isaac_{scene.get('scene_idx', i):05d}.jpg"
            else:
                img_filename = f"vlm_{i:05d}.jpg"

            records = self._scene_to_conversations(scene, img_filename)
            all_records.extend(records)

        # ── Step 3b: Mix in calibration training data ────────────────
        # Anti-hallucination patterns at 50/20/20/10 distribution.
        # These teach the VLM when to be confident vs uncertain.
        cal_yaml = Path(self.cfg["paths"]["data_root"]).expanduser().parent \
            / "cosmos_calibration_training.yaml"
        if not cal_yaml.exists():
            # Fallback: check project root
            cal_yaml = Path(__file__).resolve().parent.parent \
                / "cosmos_calibration_training.yaml"

        cal_target = max(
            int(len(all_records) * 0.20),  # at least 20% of scene Q&A
            self.vlm_cfg.get("calibration_count", 200),
        )
        mixer = CalibrationMixer(cal_yaml)
        cal_records = mixer.generate_calibration_set(cal_target)

        # Assign calibration records to random existing images so they
        # pair with real visual data during training.
        image_files = sorted(self.images_dir.glob("*.jpg")) + \
                      sorted(self.images_dir.glob("*.png"))
        if image_files:
            for rec in cal_records:
                rec["image"] = random.choice(image_files).name
        else:
            # No images yet — use placeholder name
            for idx, rec in enumerate(cal_records):
                rec["image"] = f"cal_{idx:05d}.jpg"

        all_records.extend(cal_records)
        random.shuffle(all_records)

        print(f"[VLM-Dataset] Total records after calibration mix: "
              f"{len(all_records)} "
              f"({len(cal_records)} calibration, "
              f"{len(all_records) - len(cal_records)} scene-based)")

        # ── Step 4: Write JSONL output ───────────────────────────────
        jsonl_path = self.output_dir / "vlm_finetune.jsonl"
        with open(jsonl_path, "w") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Also write a summary JSON for quick inspection
        cal_categories = {}
        for rec in cal_records:
            cat = rec.get("metadata", {}).get("calibration_category", "unknown")
            cal_categories[cat] = cal_categories.get(cat, 0) + 1

        summary = {
            "total_images": num_images,
            "total_qa_pairs": len(all_records),
            "scene_qa_pairs": len(all_records) - len(cal_records),
            "calibration_qa_pairs": len(cal_records),
            "calibration_distribution": cal_categories,
            "qa_per_image": self.qa_per_image,
            "disease_classes": self.disease_classes,
            "render_engine": "isaac_sim" if isaac_available else "procedural",
            "output_format": "ShareGPT/LLaVA conversation JSONL",
            "dataset_file": str(jsonl_path),
            "images_dir": str(self.images_dir),
        }
        with open(self.output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[VLM-Dataset] Dataset complete:")
        print(f"  Images:    {num_images}")
        print(f"  Q&A pairs: {len(all_records)}")
        print(f"  JSONL:     {jsonl_path}")
        print(f"  Format:    ShareGPT/LLaVA conversation")
        print(f"  Renderer:  {'Isaac Sim' if isaac_available else 'Procedural'}")
