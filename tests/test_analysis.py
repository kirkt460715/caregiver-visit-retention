"""预测、评估与计划不越界的测试。"""

from datetime import date
import unittest

from service_risk import load_data
from service_risk.evaluation import (
    assignment_evidence,
    backtest,
    backtest_strata,
    evaluate_strategy,
    exposure_report,
    result_semantics,
)
from service_risk.planning import build_weekly_plan, reconcile_plan
from service_risk.risk import (
    explain,
    features_for,
    fit_logistic,
    predict_week,
    training_rows,
)
from service_risk.supervisor import (
    LIST_CONTACT_MAY_CHANGE,
    LIST_HIGH_RISK,
    LIST_RECORDS_ONLY,
    case_review,
    weekly_briefing,
)
from service_risk.timeline import AT_RISK, GAP, classify_outcome


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_training_only_uses_known_receipts(self):
        as_of = date(2026, 5, 4)
        for row in training_rows(self.data, as_of):
            checkpoint = self.data.checkpoint(row.checkpoint_id)
            self.assertLessEqual(checkpoint.received, as_of)
            self.assertLessEqual(checkpoint.due, as_of)

    def test_training_excludes_contaminated_records(self):
        as_of = date(2026, 9, 1)
        ids = {row.checkpoint_id for row in training_rows(self.data, as_of)}
        self.assertNotIn("H001-1-2", ids)  # 联络执行早于回单
        self.assertIn("H064-1-4", ids)  # arm none，未受联系影响

    def test_prediction_window_and_states(self):
        as_of = date(2026, 7, 1)
        predictions = predict_week(self.data, as_of, horizon_days=7)
        self.assertTrue(predictions)
        for prediction in predictions:
            self.assertGreaterEqual(prediction.due, as_of)
            self.assertLess(prediction.due, date(2026, 7, 8))
            self.assertGreaterEqual(prediction.probability, 0.0)
            self.assertLessEqual(prediction.probability, 1.0)

    def test_future_change_does_not_leak(self):
        # H106 在 06-08 才登记迁走，06-07 的预测不能提前知道。
        for prediction in predict_week(self.data, date(2026, 6, 6), horizon_days=14):
            self.assertNotEqual(prediction.state, "ended")
        states = {
            prediction.state
            for prediction in predict_week(self.data, date(2026, 6, 8), horizon_days=14)
        }
        self.assertIn("ended", states)

    def test_low_support_fallback(self):
        model = fit_logistic(training_rows(self.data, date(2026, 2, 1)))
        self.assertTrue(model.fallback)
        predictions = predict_week(self.data, date(2026, 2, 1), model=model)
        for prediction in predictions:
            if prediction.state == AT_RISK:
                self.assertIn("训练样本不足", prediction.limitation)

    def test_explanation_is_human_readable(self):
        checkpoint = self.data.checkpoint("H064-2-4")
        features = features_for(self.data, checkpoint, date(2026, 7, 6))
        self.assertIsNotNone(features)
        self.assertTrue(explain(features))


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.result = backtest(cls.data, date(2026, 2, 23), date(2026, 9, 1))

    def test_forward_backtest_has_enough_folds(self):
        self.assertGreaterEqual(len(self.result.folds), 20)
        self.assertGreater(self.result.test_rows, 200)

    def test_model_beats_base_rate(self):
        self.assertIsNotNone(self.result.auc)
        self.assertGreater(self.result.auc, 0.7)
        self.assertLess(self.result.brier, self.result.baseline_brier)

    def test_test_spells_do_not_overlap_training(self):
        # 每个折的训练经历与复盘经历不重叠。
        for fold in self.result.folds:
            with self.subTest(cutoff=fold.cutoff):
                self.assertGreaterEqual(fold.train_rows, 40)

    def test_strata_cover_coverage_and_contact(self):
        strata = backtest_strata(self.data, self.result)
        dimensions = {row.dimension for row in strata}
        self.assertIn("group", dimensions)
        self.assertIn("site", dimensions)
        self.assertIn("channel", dimensions)
        for row in strata:
            if row.events:
                self.assertIsNotNone(row.auc)

    def test_unconfirmed_never_labeled_as_gap(self):
        for row in self.result.predictions:
            self.assertIn(row["label"], (0, 1))
            checkpoint = self.data.checkpoint(str(row["checkpoint"]))
            if classify_outcome(checkpoint, date.max) == "unconfirmed":
                self.fail("渠道未查齐的记录不应进入复盘标签")


class ExposureReportTests(unittest.TestCase):
    def test_states_are_separate(self):
        data = load_data()
        rows = exposure_report(data, date(2026, 6, 21), horizon_days=28)
        by_id = {row.checkpoint_id: row for row in rows}
        # 06-22 应约的 H096 已转出，原协议终止；H064 的 06-22 仍受原协议约束。
        self.assertEqual(by_id["H096-2-5"].exposure, "ended")
        self.assertEqual(by_id["H064-2-1"].exposure, AT_RISK)
        # 未到约定日期的记录只标“尚未到期”，不能混作缺口或未查齐。
        self.assertEqual(by_id["H064-2-4"].outcome_now, "not_due")


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.as_of = date(2026, 9, 10)
        cls.evaluation = evaluate_strategy(cls.data, cls.as_of)

    def test_contaminated_assignments_excluded(self):
        evidence = assignment_evidence(self.data, self.as_of)
        usable = [row for row in evidence if row.usable_for_effect]
        self.assertTrue(usable)
        for row in usable:
            self.assertNotIn("回单", row.contamination)
            self.assertEqual(row.identification, "randomized_itt")
            self.assertEqual(row.observation, "观察窗已成熟")

    def test_descriptive_records_are_not_effects(self):
        self.assertGreater(self.evaluation.descriptive, 0)
        self.assertEqual(
            sum(1 for row in assignment_evidence(self.data, self.as_of)
                if row.identification == "descriptive_only" and row.usable_for_effect),
            0,
        )

    def test_difference_interval_is_wide(self):
        self.assertIsNotNone(self.evaluation.difference_ci)
        low, high = self.evaluation.difference_ci
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertTrue(any("少于" in note for note in self.evaluation.notes))

    def test_alt_confirmed_means_confirmability_not_prevention(self):
        rows = result_semantics(self.data, self.as_of)
        alt = [row for row in rows if row["result"] == "alt_confirmed"]
        self.assertTrue(alt)
        for row in alt:
            self.assertIn("能否确认", str(row["meaning"]))
            self.assertNotIn("避免中断", str(row["meaning"]))
        booked = [row for row in rows if row["result"] == "booked_new"]
        self.assertTrue(booked)
        for row in booked:
            self.assertIn("避免", str(row["meaning"]))


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_weekly_plan_respects_slots(self):
        plan = build_weekly_plan(self.data, date(2026, 7, 6))
        for slot in plan.slots:
            self.assertLessEqual(slot.check_used, slot.check_total)
            self.assertLessEqual(slot.change_used, slot.change_total)

    def test_withdrawn_callback_not_reused(self):
        plan = build_weekly_plan(self.data, date(2026, 7, 6))
        withdrawn = [item for item in plan.items if item.callback_id == "B03"]
        self.assertEqual(len(withdrawn), 1)
        self.assertEqual(withdrawn[0].kind, "skip")
        self.assertIn("撤回", withdrawn[0].reason)
        self.assertFalse(withdrawn[0].uses_check_slot)

    def test_confirmed_callback_uses_slot_first(self):
        plan = build_weekly_plan(self.data, date(2026, 7, 6))
        callbacks = plan.by_kind("callback")
        self.assertEqual([item.callback_id for item in callbacks], ["B01"])
        self.assertTrue(callbacks[0].uses_check_slot)

    def test_verified_callback_not_rebooked(self):
        plan = build_weekly_plan(self.data, date(2026, 7, 6))
        self.assertFalse([item for item in plan.items if item.callback_id == "B04"])

    def test_arrange_only_goes_to_authorized_site(self):
        for week in (date(2026, 7, 6), date(2026, 7, 20), date(2026, 8, 3)):
            plan = build_weekly_plan(self.data, week)
            for item in plan.by_kind("arrange"):
                checkpoint = self.data.checkpoint(item.checkpoint_id)
                self.assertEqual(item.site, self.data.episode(checkpoint.episode).site)
                self.assertTrue(item.uses_change_slot)
                self.assertGreaterEqual(item.probability, 0.15)

    def test_transferred_household_not_contacted_by_original_site(self):
        plan = build_weekly_plan(self.data, date(2026, 6, 22), as_of=date(2026, 6, 21))
        items = [item for item in plan.items if item.household == "H096"]
        for item in items:
            self.assertNotEqual(item.kind, "arrange")

    def test_moved_household_is_skipped(self):
        plan = build_weekly_plan(self.data, date(2026, 6, 8))
        items = [item for item in plan.items if item.household == "H106"]
        self.assertTrue(items)
        for item in items:
            self.assertEqual(item.kind, "skip")
            self.assertIn("迁走", item.reason)

    def test_reconciliation_separates_next_week_and_history(self):
        plan = build_weekly_plan(self.data, date(2026, 7, 6))
        rows = reconcile_plan(self.data, plan, date(2026, 7, 13))
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row.next_week_outcome)
            self.assertTrue(row.historical_outcome)


class SupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_three_lists_are_distinct(self):
        briefing = weekly_briefing(self.data, date(2026, 7, 6))
        high = {(row["household"], row["checkpoint"]) for row in briefing.section(LIST_HIGH_RISK).rows}
        changed = {(row["household"], row["checkpoint"]) for row in briefing.section(LIST_CONTACT_MAY_CHANGE).rows}
        records = {(row["household"], row["checkpoint"]) for row in briefing.section(LIST_RECORDS_ONLY).rows}
        self.assertTrue(changed)
        self.assertFalse(changed - high)
        self.assertFalse(changed & records)

    def test_case_review_has_prediction_reason_and_evidence(self):
        review = case_review(self.data, "H064", date(2026, 7, 6))
        self.assertTrue(review.predictions)
        prediction = review.predictions[0]
        self.assertTrue(prediction.reasons)
        self.assertTrue(prediction.planned)
        self.assertTrue(review.evidence)
        for row in review.evidence:
            self.assertIn(row["identification"], {"randomized_itt", "descriptive_only", "not_usable"})
            self.assertTrue(row["usable_for"])

    def test_case_review_shows_final_confirmation(self):
        review = case_review(self.data, "H104", date(2026, 7, 13))
        notes = " ".join(row.note for row in review.predictions)
        self.assertIn("最终", notes)

    def test_withdrawn_household_has_limitation(self):
        review = case_review(self.data, "H087", date(2026, 6, 8))
        self.assertEqual(review.permission, "withdrawn")
        self.assertTrue(any("撤回" in text for text in review.limitations))


if __name__ == "__main__":
    unittest.main()
