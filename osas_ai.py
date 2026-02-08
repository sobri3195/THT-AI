"""THT AI triage for suspected OSA using snore audio + oropharyngeal photo findings.

Prototype model to prioritize PSG and early therapy decisions before definitive PSG.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import mean
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class PatientInput:
    """Input features available in first-line clinic triage."""

    age: int
    is_child: bool
    bmi: float
    neck_circumference_cm: float
    male: bool
    stopbang_score: int
    daytime_sleepiness: bool
    snore_db_mean: float
    snore_events_per_hour: float
    apnea_like_pauses_per_hour: float
    mallampati_grade: int  # 1-4
    tonsil_grade: int  # 0-4
    craniofacial_risk: bool


@dataclass(frozen=True)
class PredictionResult:
    prob_moderate_severe_ahi: float
    risk_level: str
    recommendation: str
    estimated_days_to_therapy: float
    estimated_total_diagnostic_cost_usd: float


@dataclass(frozen=True)
class CaseLabel:
    patient: PatientInput
    has_moderate_severe_ahi: bool


class OSASTriageAI:
    """Simple multimodal risk model (audio + oropharyngeal + questionnaire)."""

    def __init__(self) -> None:
        # Tuned manually as a pragmatic pre-PSG prioritization heuristic.
        self.weights = {
            "bias": -5.1,
            "stopbang": 0.45,
            "bmi": 0.07,
            "neck": 0.06,
            "male": 0.45,
            "sleepy": 0.55,
            "snore_db": 0.03,
            "snore_h": 0.015,
            "pauses_h": 0.08,
            "mallampati": 0.35,
            "tonsil": 0.28,
            "cranio": 0.65,
            "child_adjust": -0.35,
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    def predict_probability(self, patient: PatientInput) -> float:
        w = self.weights
        z = w["bias"]
        z += w["stopbang"] * patient.stopbang_score
        z += w["bmi"] * max(patient.bmi - 18.5, 0)
        z += w["neck"] * max(patient.neck_circumference_cm - 30, 0)
        z += w["male"] * int(patient.male)
        z += w["sleepy"] * int(patient.daytime_sleepiness)
        z += w["snore_db"] * max(patient.snore_db_mean - 35, 0)
        z += w["snore_h"] * patient.snore_events_per_hour
        z += w["pauses_h"] * patient.apnea_like_pauses_per_hour
        z += w["mallampati"] * max(min(patient.mallampati_grade, 4), 1)
        z += w["tonsil"] * max(min(patient.tonsil_grade, 4), 0)
        z += w["cranio"] * int(patient.craniofacial_risk)
        z += w["child_adjust"] * int(patient.is_child)
        return self._sigmoid(z)

    def recommend(self, probability: float, is_child: bool) -> tuple[str, str, float, float]:
        """Return (risk_level, recommendation, days_to_therapy, cost)."""
        if probability >= 0.8:
            risk = "Sangat Tinggi"
            recommendation = (
                "Prioritas PSG cepat (<7 hari), pertimbangkan CPAP trial dini. "
                "Jika anak dengan tonsil besar, evaluasi T&A cepat."
            )
            days = 7 if not is_child else 5
            cost = 420.0
        elif probability >= 0.6:
            risk = "Tinggi"
            recommendation = "PSG prioritas 2-4 minggu dan edukasi sleep hygiene segera."
            days = 18
            cost = 460.0
        elif probability >= 0.35:
            risk = "Sedang"
            recommendation = "Optimasi faktor risiko + PSG elektif (4-8 minggu)."
            days = 42
            cost = 520.0
        else:
            risk = "Rendah"
            recommendation = "Lanjutkan penilaian klinis; PSG bila gejala menetap/memburuk."
            days = 60
            cost = 560.0
        return risk, recommendation, days, cost

    def predict(self, patient: PatientInput) -> PredictionResult:
        p = self.predict_probability(patient)
        risk, recommendation, days, cost = self.recommend(p, patient.is_child)
        return PredictionResult(
            prob_moderate_severe_ahi=round(p, 3),
            risk_level=risk,
            recommendation=recommendation,
            estimated_days_to_therapy=days,
            estimated_total_diagnostic_cost_usd=cost,
        )


class StopBangBaseline:
    """Comparator model: questionnaire-only baseline."""

    def predict_probability(self, patient: PatientInput) -> float:
        z = -3.0 + 0.8 * patient.stopbang_score + 0.05 * max(patient.bmi - 18.5, 0)
        return 1.0 / (1.0 + exp(-z))


def binary_metrics(y_true: Sequence[bool], y_prob: Sequence[float], threshold: float = 0.5) -> dict:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")
    y_pred = [p >= threshold for p in y_prob]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))

    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(accuracy, 3),
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
    }


def compare_models(cases: Iterable[CaseLabel], threshold: float = 0.5) -> dict:
    case_list: List[CaseLabel] = list(cases)
    model = OSASTriageAI()
    baseline = StopBangBaseline()

    y_true = [c.has_moderate_severe_ahi for c in case_list]
    y_prob_mm = [model.predict_probability(c.patient) for c in case_list]
    y_prob_sb = [baseline.predict_probability(c.patient) for c in case_list]

    mm_metrics = binary_metrics(y_true, y_prob_mm, threshold)
    sb_metrics = binary_metrics(y_true, y_prob_sb, threshold)

    # Simple proxy to objective outcomes (time + diagnostic cost).
    mm_recs = [model.predict(c.patient) for c in case_list]
    avg_days = round(mean(r.estimated_days_to_therapy for r in mm_recs), 1)
    avg_cost = round(mean(r.estimated_total_diagnostic_cost_usd for r in mm_recs), 1)

    return {
        "multimodal_metrics": mm_metrics,
        "stopbang_baseline_metrics": sb_metrics,
        "avg_days_to_therapy_multimodal": avg_days,
        "avg_diagnostic_cost_usd_multimodal": avg_cost,
    }


if __name__ == "__main__":
    sample = PatientInput(
        age=46,
        is_child=False,
        bmi=32.5,
        neck_circumference_cm=42.0,
        male=True,
        stopbang_score=6,
        daytime_sleepiness=True,
        snore_db_mean=58,
        snore_events_per_hour=260,
        apnea_like_pauses_per_hour=19,
        mallampati_grade=3,
        tonsil_grade=2,
        craniofacial_risk=False,
    )

    model = OSASTriageAI()
    result = model.predict(sample)
    print("=== Hasil Triage OSAS (Multimodal AI) ===")
    print(f"Probabilitas AHI sedang-berat : {result.prob_moderate_severe_ahi}")
    print(f"Kategori risiko               : {result.risk_level}")
    print(f"Rekomendasi                   : {result.recommendation}")
    print(f"Estimasi waktu ke terapi      : {result.estimated_days_to_therapy} hari")
    print(f"Estimasi biaya diagnosis      : ${result.estimated_total_diagnostic_cost_usd}")
