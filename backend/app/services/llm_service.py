"""
Multi-LLM explanation service.
Fallback chain: Groq (llama-3.1-8b-instant) → Anthropic Claude → OpenAI → rule-based fallback
"""
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.config import settings
from app.models.schemas import PatientInput

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    clinical_reasoning: str
    model_used: str


def _build_prompt(inp: PatientInput, severity: str) -> str:
    comorbidities = []
    if inp.diabetes:           comorbidities.append("diabetes")
    if inp.copd:               comorbidities.append("COPD")
    if inp.asthma:             comorbidities.append("asthma")
    if inp.immunocompromised:  comorbidities.append("immunosuppression")
    if inp.hypertension:       comorbidities.append("hypertension")
    if inp.cardiovascular:     comorbidities.append("cardiovascular disease")
    if inp.obesity:            comorbidities.append("obesity")
    if inp.renal_chronic:      comorbidities.append("chronic renal disease")
    if inp.tobacco:            comorbidities.append("tobacco use")

    risk_str = ", ".join(comorbidities) if comorbidities else "none"
    clinical_str = "hospitalized" if inp.hospitalized else "outpatient"
    pneumonia_str = "yes" if inp.pneumonia else "no"

    return f"""Patient: {inp.age}-year-old {inp.sex}.
Risk factors present: {risk_str}.
Clinical status: {clinical_str}, pneumonia: {pneumonia_str}.
COVID-19 severity assessment: {severity}.

In 2 sentences explain why this patient is classified as {severity} severity, referencing their specific age and risk factors.
Then state the recommended next steps clearly.

Respond with JSON only:
{{"clinical_reasoning": "Your explanation here"}}"""


class LLMService:
    """Groq → Anthropic → OpenAI fallback chain for LLM explanations."""

    def explain(self, inp: PatientInput, severity: str) -> LLMResult:
        prompt = _build_prompt(inp, severity)

        if settings.groq_api_key:
            result = self._try_groq(prompt)
            if result:
                return result

        if settings.anthropic_api_key:
            result = self._try_anthropic(prompt)
            if result:
                return result

        if settings.openai_api_key:
            result = self._try_openai(prompt)
            if result:
                return result

        return self._rule_based_fallback(inp, severity)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> Optional[Dict]:
        text = text.strip()
        if "```" in text:
            parts = [p for p in text.split("```") if p.strip()]
            if parts:
                text = parts[0].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
        return None

    # ── Provider attempts ─────────────────────────────────────────────────────

    def _try_groq(self, prompt: str) -> Optional[LLMResult]:
        try:
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            data = self._parse_json(response.choices[0].message.content)
            if data:
                return LLMResult(
                    clinical_reasoning=data.get("clinical_reasoning", ""),
                    model_used=f"groq/{settings.groq_model}",
                )
        except Exception as e:
            print(f"GROQ ERROR: {e}", flush=True)
        return None

    def _try_anthropic(self, prompt: str) -> Optional[LLMResult]:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            data = self._parse_json(response.content[0].text)
            if data:
                return LLMResult(
                    clinical_reasoning=data.get("clinical_reasoning", ""),
                    model_used=f"anthropic/{settings.anthropic_model}",
                )
        except Exception as e:
            logger.warning(f"Anthropic failed: {e}")
        return None

    def _try_openai(self, prompt: str) -> Optional[LLMResult]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            data = self._parse_json(response.choices[0].message.content)
            if data:
                return LLMResult(
                    clinical_reasoning=data.get("clinical_reasoning", ""),
                    model_used=f"openai/{settings.openai_model}",
                )
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")
        return None

    # ── Rule-based fallback ────────────────────────────────────────────────────

    def _rule_based_fallback(self, inp: PatientInput, severity: str) -> LLMResult:
        comorbidities = []
        if inp.diabetes:          comorbidities.append("diabetes")
        if inp.hypertension:      comorbidities.append("hypertension")
        if inp.cardiovascular:    comorbidities.append("cardiovascular disease")
        if inp.obesity:           comorbidities.append("obesity")
        if inp.copd:              comorbidities.append("COPD")
        if inp.immunocompromised: comorbidities.append("immunosuppression")
        if inp.renal_chronic:     comorbidities.append("chronic renal disease")

        risk_str = ", ".join(comorbidities[:3]) if comorbidities else "no significant comorbidities"
        age_desc = "elderly" if inp.age > 60 else "adult"

        messages = {
            "Critical": (
                f"This {inp.age}-year-old {age_desc} patient with {risk_str} presents critical COVID-19 "
                f"indicators requiring immediate ICU intervention. "
                "Call emergency services now — do not delay."
            ),
            "Severe": (
                f"This {inp.age}-year-old patient with {risk_str} requires emergency department care "
                "due to severe COVID-19 progression. Go to the emergency department immediately."
            ),
            "Moderate": (
                f"This {inp.age}-year-old patient with {risk_str} has moderate COVID-19 requiring "
                "medical evaluation within 24 hours. Ask your doctor about antiviral eligibility."
            ),
            "Mild": (
                f"This {inp.age}-year-old patient with {risk_str} has mild COVID-19 consistent with "
                "home management. Isolate for 10 days, rest, and monitor oxygen saturation."
            ),
            "No_COVID": (
                f"This {inp.age}-year-old patient's profile ({risk_str}) does not meet COVID-19 "
                "severity criteria. Continue monitoring and retest if symptoms develop."
            ),
        }
        return LLMResult(
            clinical_reasoning=messages.get(severity, "Please consult a healthcare provider."),
            model_used="rule-based-fallback",
        )
