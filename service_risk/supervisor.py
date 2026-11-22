"""服务督导的个案与周报视图。

督导需要同时看到四件事，而不是一张混在一起的名单：

1. 各时点的中断风险和适用范围（只对仍受原协议约束的应约日给出风险）；
2. 按覆盖与联络策略回看预警能否找准服务缺口（前向回测，按经历留出）；
3. 高风险、下一次联系可能改变安排、仅适合核查记录，是三个不同的名单；
4. 每条证据能用于评价安排，还是只能描述已联络个案。

低支持时明确给出限制，而不是照常给出精确数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .data import PolicyData
from .evaluation import (
    StrategyEvaluation,
    assignment_evidence,
    evaluate_strategy,
    exposure_report,
    result_semantics,
)
from .planning import WeeklyPlan, build_weekly_plan
from .risk import (
    RiskModel,
    band_for,
    explain,
    features_for,
    fit_logistic,
    predict_week,
    training_rows,
)
from .timeline import (
    AT_RISK,
    classify_outcome,
    exposure_at,
    exposure_for_episode,
    final_outcome,
    outcome_label,
    spell_ids,
)

# 三个名单的固定标识，供下游按标识而不是按标题取用。
LIST_HIGH_RISK = "high_risk"
LIST_CONTACT_MAY_CHANGE = "contact_may_change"
LIST_RECORDS_ONLY = "records_only"


@dataclass(frozen=True)
class CasePrediction:
    checkpoint_id: str
    due: date
    probability: float | None
    band: str
    state: str
    reasons: tuple[str, ...]
    limitation: str
    planned: str
    planned_reason: str
    outcome_now: str
    outcome_final: str
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_id,
            "due": self.due.isoformat(),
            "probability": None if self.probability is None else round(self.probability, 4),
            "band": self.band,
            "state": self.state,
            "reasons": list(self.reasons),
            "limitation": self.limitation,
            "planned": self.planned,
            "planned_reason": self.planned_reason,
            "outcome_now": self.outcome_now,
            "outcome_final": self.outcome_final,
            "note": self.note,
        }


@dataclass(frozen=True)
class CaseReview:
    household: str
    as_of: date
    channel: str
    permission: str
    site: str
    episode_timeline: tuple[dict[str, object], ...]
    predictions: tuple[CasePrediction, ...]
    evidence: tuple[dict[str, object], ...]
    spell_ids: tuple[str, ...]
    limitations: tuple[str, ...]


def _protocol_timeline(data: PolicyData, household: str, as_of: date) -> tuple[dict[str, object], ...]:
    """列出该家庭每段协议及已登记的变更，并标出在当前时点是否可见。"""
    rows: list[dict[str, object]] = []
    for episode in data.episodes_by_household.get(household, ()):
        changes = data.changes_by_episode.get(episode.episode, ())
        base = {
            "episode": episode.episode,
            "site": episode.site,
            "group": episode.group,
            "begins": episode.begins.isoformat(),
            "ends": episode.ends.isoformat(),
        }
        if not changes:
            rows.append(
                {
                    **base,
                    "change": "",
                    "change_starts": "",
                    "change_ends": "",
                    "provider": "",
                    "recorded_on": "",
                    "known_at_as_of": True,
                    "current_state": exposure_for_episode(
                        data, episode, max(as_of, episode.begins), as_of
                    ).state,
                }
            )
            continue
        for change in changes:
            rows.append(
                {
                    **base,
                    "change": change.status,
                    "change_starts": change.starts.isoformat(),
                    "change_ends": change.ends.isoformat() if change.ends else "",
                    "provider": change.new_provider,
                    "recorded_on": change.recorded_on.isoformat(),
                    "known_at_as_of": change.recorded_on <= as_of,
                    "current_state": exposure_for_episode(
                        data, episode, max(as_of, change.starts), as_of
                    ).state,
                }
            )
    return tuple(rows)


def case_review(
    data: PolicyData,
    household: str,
    as_of: date,
    *,
    horizon_days: int = 7,
    model: RiskModel | None = None,
) -> CaseReview:
    """按个案查看预测、联系安排、最终确认了什么、证据能否评价安排。"""
    if model is None:
        model = fit_logistic(training_rows(data, as_of))
    week_predictions = predict_week(data, as_of, horizon_days=7, model=model)
    predictions = tuple(row for row in week_predictions if row.household == household)
    plan = build_weekly_plan(data, as_of, as_of=as_of, predictions=week_predictions)
    plan_by_checkpoint = {item.checkpoint_id: item for item in plan.items}
    mapping = spell_ids(data, as_of)

    rows: list[CasePrediction] = []
    # 未来已排定的应约日也列出；即使当前已暂停/转出，主管仍要看到协议原定日期。
    checkpoints = [
        row
        for row in data.checkpoints_for_household(household)
        if as_of <= row.due < as_of + timedelta(days=horizon_days)
    ]
    for checkpoint in checkpoints:
        exposure = exposure_at(data, checkpoint, as_of)
        item = plan_by_checkpoint.get(checkpoint.id)
        probability = None
        band = "not_at_risk"
        reasons: tuple[str, ...] = ()
        limitation = exposure.reason if exposure.state != AT_RISK else ""
        if exposure.state == AT_RISK:
            features = features_for(data, checkpoint, as_of, spell_map=mapping)
            if features is not None:
                probability = model.predict(features)
                band = band_for(probability)
                reasons = explain(features)
            if model.fallback:
                limitation = "训练样本不足，暂用整体缺口率"
            elif probability is not None and probability >= 0.30:
                limitation = "高风险名单只表示优先核查，不表示一定缺服务"
        now = classify_outcome(checkpoint, as_of)
        final = final_outcome(checkpoint)
        rows.append(
            CasePrediction(
                checkpoint_id=checkpoint.id,
                due=checkpoint.due,
                probability=probability,
                band=band,
                state=exposure.state,
                reasons=reasons,
                limitation=limitation,
                planned=item.kind if item else "not_planned",
                planned_reason=(
                    item.reason
                    if item
                    else (
                        "本周计划把它列为无需联系：应约日不在原协议风险中"
                        if exposure.state != AT_RISK
                        else "不在本周计划范围内"
                    )
                ),
                outcome_now=now,
                outcome_final=final,
                note=(
                    f"{exposure.reason}；该应约日不作为原协议的中断风险，"
                    f"回单只用于核对交接"
                    if exposure.state != AT_RISK
                    else (
                        f"按 {as_of.isoformat()} 只能确认到“{outcome_label(now)}”；"
                        f"复盘口径最终为“{outcome_label(final)}”"
                        if now != final
                        else outcome_label(final)
                    )
                ),
            )
        )

    evidence_rows: list[dict[str, object]] = []
    outreach_by_id = {row.id: row for row in data.outreach}
    for evidence in assignment_evidence(data, as_of):
        if evidence.household != household:
            continue
        outreach = outreach_by_id[evidence.outreach_id]
        meaning = {
            "alt_confirmed": "改变了结果能否确认：由无服务改记为有替代安排覆盖",
            "booked_new": "联络后新安排了服务，可能真的避免了中断",
            "no_change": "没有改变安排或证明",
        }[outreach.result]
        usable = {
            "randomized_itt": "可进入意向性比较",
            "descriptive_only": "只能描述这个已联络个案",
            "not_usable": "不能用于评价联络策略",
        }[evidence.identification]
        evidence_rows.append(
            {
                "outreach": outreach.id,
                "checkpoint": evidence.checkpoint_id,
                "design": outreach.design,
                "arm": outreach.arm,
                "action_day": outreach.action_day.isoformat() if outreach.action_day else "",
                "result": outreach.result,
                "meaning": meaning,
                "identification": evidence.identification,
                "observation": evidence.observation,
                "usable_for": usable,
                "reason": evidence.reason,
                "target_outcome": evidence.target_outcome,
                "next_checkpoint": evidence.next_checkpoint_id,
                "next_outcome": evidence.next_outcome,
            }
        )

    permission = data.permission(household)
    spell_set = sorted(
        {
            mapping[row.id]
            for row in data.checkpoints_for_household(household)
            if mapping.get(row.id)
        }
    )
    limitations: list[str] = []
    if model.fallback:
        limitations.append("训练样本不足，预测只能作描述性参考。")
    if permission is None:
        limitations.append("没有联络授权记录，计划不会主动联系该家庭。")
    elif not permission.active:
        limitations.append("家庭已撤回联络授权，计划不会主动联系。")
    timeline = _protocol_timeline(data, household, as_of)
    if any(row["known_at_as_of"] is False for row in timeline):
        limitations.append("存在登记日晚于当前时点的变更，当前时点看不到，不能提前影响判断。")
    return CaseReview(
        household=household,
        as_of=as_of,
        channel=permission.channel if permission else "",
        permission=permission.status if permission else "missing",
        site=permission.site if permission else "",
        episode_timeline=timeline,
        predictions=tuple(rows),
        evidence=tuple(evidence_rows),
        spell_ids=tuple(spell_set),
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class BriefingSection:
    key: str
    title: str
    note: str
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class WeeklyBriefing:
    week: date
    as_of: date
    plan: WeeklyPlan
    sections: tuple[BriefingSection, ...]
    strategy: StrategyEvaluation
    support: tuple[str, ...]

    def section(self, key: str) -> BriefingSection:
        """按固定标识取名单，避免标题措辞变化影响下游。"""
        for section in self.sections:
            if section.key == key:
                return section
        raise KeyError(key)


def weekly_briefing(
    data: PolicyData,
    week: date,
    *,
    as_of: date | None = None,
) -> WeeklyBriefing:
    """生成一周的督导简报。

    三个名单分开：

    * ``高风险``：预测概率较高、仍受原协议约束的应约日；
    * ``下一次联系可能改变安排``：本周计划中直接改约的个案；
    * ``仅适合核查记录``：因样本不足、额度不足、授权撤回、无处理权站点等原因，
      只能查证而不宜主动改约的个案。
    """
    as_of = as_of or week
    model = fit_logistic(training_rows(data, as_of))
    predictions = predict_week(data, as_of, horizon_days=7, model=model)
    plan = build_weekly_plan(data, week, as_of=as_of, predictions=predictions)
    prediction_by_checkpoint = {row.checkpoint_id: row for row in predictions}

    high_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []
    records_rows: list[dict[str, object]] = []
    for prediction in predictions:
        if prediction.state != AT_RISK:
            continue
        item = next(
            (candidate for candidate in plan.items if candidate.checkpoint_id == prediction.checkpoint_id),
            None,
        )
        base = {
            "checkpoint": prediction.checkpoint_id,
            "household": prediction.household,
            "due": prediction.due.isoformat(),
            "probability": round(prediction.probability, 4),
            "band": prediction.band,
            "planned": item.kind if item else "not_planned",
            "channel": item.channel if item else "",
            "site": item.site if item else "",
            "reason": item.reason if item else "",
            "limitation": prediction.limitation,
        }
        if prediction.band in ("high", "moderate"):
            high_rows.append(base)
        if item is not None and item.kind == "arrange":
            change_rows.append(base)
        elif item is not None and (
            item.kind in ("document", "skip")
            and ("额度" in item.reason or "未达到直接改约" in item.reason or "撤回" in item.reason)
        ):
            records_rows.append(base)

    evidence = assignment_evidence(data, as_of)
    strategy = evaluate_strategy(data, as_of)
    support: list[str] = []
    if model.fallback:
        support.append("预测模型处于低支持：训练样本不足，三个名单都只能作核查参考。")
    if strategy.population < 40:
        support.append(
            f"可比较的联络记录只有 {strategy.population} 条，观察窗区间很宽，"
            "不能据此断言哪种联络策略更有效。"
        )
    if strategy.contaminated >= 40:
        support.append(
            f"有 {strategy.contaminated} 条联络执行早于回单，结果可能被联系改变，"
            "这些记录不能进入效果比较。"
        )
    unconfirmed_fraction = _unconfirmed_fraction(data, as_of)
    if unconfirmed_fraction is not None and unconfirmed_fraction > 0.15:
        support.append(
            f"截至 {as_of.isoformat()} 已到期记录中有 {unconfirmed_fraction:.0%} 渠道未查齐；"
            "这些既不算缺口，也不算有服务。"
        )

    sections = (
        BriefingSection(
            key=LIST_HIGH_RISK,
            title="高风险与次高风险",
            note=(
                "按当前可见记录预测中断概率较高的应约日，分 high 与 moderate 两档；"
                "仍受原协议约束才列入。列入不等于已发生缺口。"
            ),
            rows=tuple(sorted(high_rows, key=lambda row: (-float(row["probability"]), str(row["due"])))),
        ),
        BriefingSection(
            key=LIST_CONTACT_MAY_CHANGE,
            title="下一次联系可能改变安排",
            note=(
                "本周计划中直接改约的个案：站点对当前协议有处理权且本周有改约额度。"
                "只有这一类联系可能真正避免中断。"
            ),
            rows=tuple(change_rows),
        ),
        BriefingSection(
            key=LIST_RECORDS_ONLY,
            title="仅适合核查记录",
            note=(
                "额度不足、未达改约阈值、授权撤回或没有处理权站点：只宜查证记录，"
                "不应把联系本身当成干预效果。"
            ),
            rows=tuple(records_rows),
        ),
        BriefingSection(
            key="evidence_strength",
            title="证据强度",
            note=(
                "登记分派且执行不早于回单的记录可进入意向性比较；"
                "其余只能描述已联络个案，或完全不能用于评价。"
            ),
            rows=tuple(
                {
                    "outreach": row.outreach_id,
                    "household": row.household,
                    "arm": row.arm,
                    "identification": row.identification,
                    "observation": row.observation,
                    "reason": row.reason,
                }
                for row in evidence
                if row.household in {prediction.household for prediction in predictions}
            ),
        ),
    )
    return WeeklyBriefing(
        week=week,
        as_of=as_of,
        plan=plan,
        sections=sections,
        strategy=strategy,
        support=tuple(support),
    )


def _unconfirmed_fraction(data: PolicyData, as_of: date) -> float | None:
    due = [
        row
        for row in data.checkpoints
        if row.due <= as_of and row.received <= as_of
    ]
    if not due:
        return None
    unconfirmed = sum(1 for row in due if classify_outcome(row, as_of) == "unconfirmed")
    return unconfirmed / len(due)


def strategy_review(data: PolicyData, as_of: date) -> dict[str, object]:
    """把策略评估压成督导可读的摘要，并明确哪些是效果证据、哪些只是描述。"""
    evaluation = evaluate_strategy(data, as_of)
    semantics = result_semantics(data, as_of)
    bookkeeping: dict[str, int] = {}
    for row in assignment_evidence(data, as_of):
        bookkeeping[row.identification] = bookkeeping.get(row.identification, 0) + 1
    by_result: dict[str, int] = {}
    for row in semantics:
        by_result[row["result"]] = by_result.get(row["result"], 0) + 1
    return {
        "as_of": as_of.isoformat(),
        "population": evaluation.population,
        "difference": evaluation.difference,
        "difference_ci": evaluation.difference_ci,
        "cells": [
            {
                "arm": cell.arm,
                "assignments": cell.assignments,
                "eligible": cell.eligible,
                "next_gaps": cell.next_gaps,
                "observed_rate": cell.observed_rate,
                "weighted_rate": cell.weighted_rate,
            }
            for cell in evaluation.cells
        ],
        "identification": bookkeeping,
        "results": by_result,
        "notes": evaluation.notes,
    }
