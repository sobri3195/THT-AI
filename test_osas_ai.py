import unittest

from osas_ai import CaseLabel, OSASTriageAI, PatientInput, compare_models


class TestOSASTriageAI(unittest.TestCase):
    def test_high_risk_patient(self):
        p = PatientInput(
            age=52,
            is_child=False,
            bmi=35,
            neck_circumference_cm=45,
            male=True,
            stopbang_score=7,
            daytime_sleepiness=True,
            snore_db_mean=60,
            snore_events_per_hour=300,
            apnea_like_pauses_per_hour=22,
            mallampati_grade=4,
            tonsil_grade=3,
            craniofacial_risk=False,
        )
        result = OSASTriageAI().predict(p)
        self.assertGreaterEqual(result.prob_moderate_severe_ahi, 0.8)

    def test_compare_models_structure(self):
        cases = [
            CaseLabel(
                patient=PatientInput(
                    age=45,
                    is_child=False,
                    bmi=31,
                    neck_circumference_cm=42,
                    male=True,
                    stopbang_score=6,
                    daytime_sleepiness=True,
                    snore_db_mean=55,
                    snore_events_per_hour=240,
                    apnea_like_pauses_per_hour=16,
                    mallampati_grade=3,
                    tonsil_grade=2,
                    craniofacial_risk=False,
                ),
                has_moderate_severe_ahi=True,
            ),
            CaseLabel(
                patient=PatientInput(
                    age=29,
                    is_child=False,
                    bmi=23,
                    neck_circumference_cm=34,
                    male=False,
                    stopbang_score=1,
                    daytime_sleepiness=False,
                    snore_db_mean=38,
                    snore_events_per_hour=65,
                    apnea_like_pauses_per_hour=1,
                    mallampati_grade=1,
                    tonsil_grade=1,
                    craniofacial_risk=False,
                ),
                has_moderate_severe_ahi=False,
            ),
        ]
        summary = compare_models(cases)
        self.assertIn("multimodal_metrics", summary)
        self.assertIn("stopbang_baseline_metrics", summary)


if __name__ == "__main__":
    unittest.main()
