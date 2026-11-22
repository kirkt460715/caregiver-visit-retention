"""时点口径、风险经历与数据校验的测试。"""

from datetime import date
import unittest

from service_risk import load_data
from service_risk.timeline import (
    AT_RISK,
    COVERED,
    ENDED,
    GAP,
    NOT_DUE,
    SERVED,
    SUSPENDED,
    UNCONFIRMED,
    authority_site,
    classify_outcome,
    contamination_reason,
    exposure_at,
    gap_label,
    spell_ids,
    spells,
)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_data_is_consistent(self):
        self.assertEqual(self.data.validate(), [])

    def test_every_episode_has_six_checkpoints(self):
        for episode in self.data.episodes:
            with self.subTest(episode=episode.episode):
                self.assertEqual(
                    len(self.data.checkpoints_by_episode[episode.episode]), 6
                )


class ExposureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_future_change_is_invisible(self):
        checkpoint = self.data.checkpoint("H106-1-4")
        before = exposure_at(self.data, checkpoint, date(2026, 6, 7))
        after = exposure_at(self.data, checkpoint, date(2026, 6, 8))
        self.assertEqual(before.state, AT_RISK)
        self.assertEqual(after.state, ENDED)

    def test_change_applies_by_actual_dates_not_registration(self):
        # H064-2 的暂住从 2026-06-07 生效，06-08 才登记；应约日 06-15 落在窗口内。
        checkpoint = self.data.checkpoint("H064-2-0")
        self.assertEqual(exposure_at(self.data, checkpoint, date(2026, 6, 1)).state, AT_RISK)
        self.assertEqual(exposure_at(self.data, checkpoint, date(2026, 6, 9)).state, SUSPENDED)

    def test_suspension_window_ends(self):
        # 暂住 06-17 结束，06-22 应约已恢复原协议风险，且窗口左闭右开。
        later = self.data.checkpoint("H064-2-1")
        self.assertEqual(exposure_at(self.data, later, date.max).state, AT_RISK)
        self.assertEqual(
            exposure_at(self.data, self.data.checkpoint("H104-2-1"), date.max).state, COVERED
        )

    def test_moved_household_has_no_original_risk_afterwards(self):
        for checkpoint_id in ("H106-1-2", "H106-1-3", "H106-1-4", "H106-1-5"):
            with self.subTest(checkpoint=checkpoint_id):
                self.assertEqual(
                    exposure_at(self.data, self.data.checkpoint(checkpoint_id), date.max).state,
                    ENDED,
                )
        self.assertEqual(
            exposure_at(self.data, self.data.checkpoint("H106-1-1"), date.max).state,
            AT_RISK,
        )

    def test_transfer_ends_original_agreement(self):
        checkpoint = self.data.checkpoint("H096-2-5")
        self.assertEqual(exposure_at(self.data, checkpoint, date.max).state, ENDED)
        self.assertEqual(
            authority_site(self.data, "H096", checkpoint.due, date.max)[0], "S3"
        )

    def test_confirmed_alternate_covers_only_its_dates(self):
        covered = self.data.checkpoint("H110-1-5")
        self.assertEqual(exposure_at(self.data, covered, date.max).state, COVERED)
        before = self.data.checkpoint("H110-1-4")
        self.assertEqual(exposure_at(self.data, before, date.max).state, AT_RISK)
        self.assertEqual(
            authority_site(self.data, "H110", covered.due, date.max)[0], "S4"
        )

    def test_moved_household_has_no_authority_site(self):
        site, _ = authority_site(self.data, "H106", date(2026, 6, 15), date.max)
        self.assertIsNone(site)


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_three_states_are_separate(self):
        self.assertEqual(classify_outcome(self.data.checkpoint("H001-1-0"), date(2026, 2, 8)), NOT_DUE)
        self.assertEqual(
            classify_outcome(self.data.checkpoint("H001-1-0"), date(2026, 2, 10)),
            "awaiting_receipt",
        )
        self.assertEqual(classify_outcome(self.data.checkpoint("H001-1-0"), date(2026, 2, 11)), SERVED)
        self.assertEqual(
            classify_outcome(self.data.checkpoint("H001-1-3"), date(2026, 3, 7)), UNCONFIRMED
        )
        self.assertEqual(
            classify_outcome(self.data.checkpoint("H001-1-2"), date(2026, 3, 1)), GAP
        )

    def test_unconfirmed_is_not_a_gap(self):
        checkpoint = self.data.checkpoint("H001-1-3")
        self.assertEqual(classify_outcome(checkpoint, date.max), UNCONFIRMED)
        self.assertEqual(gap_label(UNCONFIRMED), UNCONFIRMED)
        self.assertNotEqual(gap_label(UNCONFIRMED), "gap")

    def test_preexisting_alternate_is_not_a_gap(self):
        checkpoint = self.data.checkpoint("H004-1-2")
        self.assertEqual(classify_outcome(checkpoint, date.max), "covered_elsewhere")
        self.assertEqual(gap_label("covered_elsewhere"), "no_gap")


class ContaminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_action_before_receipt_is_flagged(self):
        checkpoint = self.data.checkpoint("H001-1-2")
        self.assertIn("回单", contamination_reason(self.data, checkpoint))

    def test_arm_none_is_clean(self):
        for checkout in self.data.checkpoints:
            outreach = self.data.outreach_by_checkpoint.get(checkout.id)
            if outreach and outreach.arm == "none":
                self.assertEqual(contamination_reason(self.data, checkout), "")


class SpellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()

    def test_moved_period_is_not_in_a_spell(self):
        mapping = spell_ids(self.data)
        self.assertTrue(mapping["H106-1-1"])
        for checkpoint_id in ("H106-1-2", "H106-1-3", "H106-1-4", "H106-1-5"):
            self.assertEqual(mapping[checkpoint_id], "")

    def test_suspension_splits_spell(self):
        mapping = spell_ids(self.data)
        self.assertEqual(mapping["H064-1-5"], "H064:H064-1:1")
        self.assertEqual(mapping["H064-2-0"], "")
        self.assertEqual(mapping["H064-2-1"], "H064:H064-2:1")

    def test_reopened_agreement_is_a_new_spell(self):
        mapping = spell_ids(self.data)
        self.assertNotEqual(mapping["H008-1-5"], mapping["H008-2-0"])
        self.assertEqual(len(spells(self.data)), len({v for v in mapping.values() if v}))

    def test_each_episode_is_contiguous(self):
        grouped = spells(self.data)
        total = sum(len(spell.checkpoint_ids) for spell in grouped.values())
        self.assertEqual(total, sum(1 for value in spell_ids(self.data).values() if value))


if __name__ == "__main__":
    unittest.main()
