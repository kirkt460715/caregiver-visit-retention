"""每周联系计划。

计划遵守：

* 家庭同意的方式（``permissions.channel``；``withdrawn`` 不可联系）；
* 不同机构确认的联络优先权（已转出/已由其他机构替代的家庭由承接机构处理，
  原站点避免重复联系）；
* 班期额度（``capacity`` 的查证与改约次数）；
* 已约定的回拨时间（``callback.due``）。

直接改约（arrange）只送给对当前协议有处理权、且存在可落实服务的站点。接到
明确撤回的回拨不能被当作新的联系机会；已经核验的回拨也不重复占用名额。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .data import Checkpoint, PolicyData
from .risk import Prediction, predict_week
from .timeline import (
    AT_RISK,
    authority_site,
    exposure_at,
    exposure_for_household,
    spell_ids,
)

KIND_CALLBACK = "callback"
KIND_DOCUMENT = "document"
KIND_ARRANGE = "arrange"
KIND_SKIP = "skip"

# 达到该预测概率才考虑直接改约；低于此值只查证。
ARRANGE_THRESHOLD = 0.15


@dataclass(frozen=True)
class PlanItem:
    household: str
    checkpoint_id: str
    episode: str
    due: date
    week: date
    kind: str
    site: str | None
    channel: str
    probability: float | None
    uses_check_slot: bool
    uses_change_slot: bool
    callback_id: str
    reason: str
    next_week_check: str

    @property
    def is_contact(self) -> bool:
        return self.kind in (KIND_CALLBACK, KIND_DOCUMENT, KIND_ARRANGE)

    def as_dict(self) -> dict[str, object]:
        return {
            "household": self.household,
            "checkpoint": self.checkpoint_id,
            "due": self.due.isoformat(),
            "week": self.week.isoformat(),
            "kind": self.kind,
            "site": self.site or "",
            "channel": self.channel,
            "probability": None if self.probability is None else round(self.probability, 4),
            "check_slot": self.uses_check_slot,
            "change_slot": self.uses_change_slot,
            "callback": self.callback_id,
            "reason": self.reason,
            "next_week_check": self.next_week_check,
        }


@dataclass(frozen=True)
class SlotLedger:
    site: str
    week: date
    check_total: int
    check_used: int
    change_total: int
    change_used: int


@dataclass(frozen=True)
class WeeklyPlan:
    week: date
    as_of: date
    items: tuple[PlanItem, ...]
    slots: tuple[SlotLedger, ...]
    notes: tuple[str, ...]

    def by_kind(self, kind: str) -> tuple[PlanItem, ...]:
        return tuple(item for item in self.items if item.kind == kind)


def _next_due_text(data: PolicyData, checkpoint: Checkpoint) -> str:
    rows = data.checkpoints_by_episode.get(checkpoint.episode, ())
    for index, row in enumerate(rows):
        if row.id == checkpoint.id and index + 1 < len(rows):
            following = rows[index + 1]
            return f"下周核对 {following.due.isoformat()} 的回单（渠道：{_channel_text(data, following.id)}）"
    return "本协议最后一次应约；下周核对最终回单是否可确认"


def _channel_text(data: PolicyData, checkpoint_id: str) -> str:
    checkpoint = data.checkpoint(checkpoint_id)
    permission = data.permission(data.household_of(checkpoint))
    return permission.channel if permission else "未登记"


def build_weekly_plan(
    data: PolicyData,
    week: date,
    *,
    as_of: date | None = None,
    predictions: tuple[Prediction, ...] | None = None,
) -> WeeklyPlan:
    """为 ``week`` 所在周生成联系计划。

    ``as_of`` 默认取周一；预测只用当天能得到的记录。回拨按约定时间优先占名额。
    """
    as_of = as_of or week
    if predictions is None:
        predictions = predict_week(data, as_of, horizon_days=7)
    probability_by_checkpoint = {
        prediction.checkpoint_id: prediction.probability
        for prediction in predictions
        if prediction.state == AT_RISK
    }

    ledger: dict[str, dict[str, int]] = {}
    for capacity in data.capacity:
        if capacity.week == week:
            ledger[capacity.site] = {
                "check_total": capacity.check_slots,
                "check_used": 0,
                "change_total": capacity.change_slots,
                "change_used": 0,
            }

    items: list[PlanItem] = []
    notes: list[str] = []
    handled: set[str] = set()

    # 1) 已约定的回拨先占名额；撤回的不得当作新机会。
    for callback in data.callbacks:
        if not (week <= callback.due < week + timedelta(days=7)):
            continue
        # 回拨按约定时间落在本周；应约记录按回拨当天的日期找，而不是整周第一条。
        checkpoint = next(
            (
                row
                for row in data.checkpoints_for_household(callback.household)
                if row.due == callback.due
            ),
            None,
        )
        channel = (data.permission(callback.household).channel
                   if data.permission(callback.household) else "未登记")
        if callback.status == "withdrawn":
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=None,
                    channel=channel,
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason="家庭已明确撤回联络授权，待回拨作废，不能当作新机会",
                    next_week_check="不再联系；如另有书面重新授权再评估",
                )
            )
            if checkpoint:
                handled.add(checkpoint.id)
            continue
        if callback.status == "verified":
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=None,
                    channel=channel,
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason="回拨已核验，记录已确认，不重复占用名额",
                    next_week_check="直接进入下周回单核对",
                )
            )
            if checkpoint:
                handled.add(checkpoint.id)
            continue
        # confirmed：占用一个查证名额；若当天没有协议覆盖，仍保留承诺的回拨。
        permission = data.permission(callback.household)
        exposure = exposure_for_household(data, callback.household, callback.due, as_of)
        if permission is not None and not permission.active:
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=callback.site,
                    channel=permission.channel,
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason="家庭已撤回联络授权，即使回拨仍登记为已约定也不能主动联系",
                    next_week_check="不再联系；如另有书面重新授权再评估",
                )
            )
            if checkpoint:
                handled.add(checkpoint.id)
            continue
        authority, basis = authority_site(data, callback.household, callback.due, as_of)
        if authority is None:
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=callback.site,
                    channel=permission.channel if permission else "未登记",
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason=f"没有可落实服务的站点（{basis}），回拨无法按原约定执行",
                    next_week_check="确认迁住或转出是否已登记完整",
                )
            )
            if checkpoint:
                handled.add(checkpoint.id)
            continue
        if authority != callback.site:
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=callback.site,
                    channel=permission.channel if permission else "未登记",
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason=(
                        f"回拨登记在 {callback.site}，但当前有处理权的是 {authority}"
                        f"（{basis}）；应避免重复联系"
                    ),
                    next_week_check=f"由 {authority} 在交接后反馈结果",
                )
            )
            if checkpoint:
                handled.add(checkpoint.id)
            continue
        if callback.site not in ledger:
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=callback.site,
                    channel=channel,
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason="本周没有该站点的余量资料，已承诺的回拨无法落实，需主管协调",
                    next_week_check="补录该周余量后重新排期",
                )
            )
            continue
        slots = ledger[callback.site]
        if slots["check_used"] >= slots["check_total"]:
            items.append(
                PlanItem(
                    household=callback.household,
                    checkpoint_id=checkpoint.id if checkpoint else "",
                    episode=checkpoint.episode if checkpoint else "",
                    due=callback.due,
                    week=week,
                    kind=KIND_SKIP,
                    site=callback.site,
                    channel=channel,
                    probability=None,
                    uses_check_slot=False,
                    uses_change_slot=False,
                    callback_id=callback.id,
                    reason="已承诺的回拨与本周期查证额度冲突，需主管调配",
                    next_week_check="在约定时间内重新安排回拨，避免拖过下周",
                )
            )
            continue
        slots["check_used"] += 1
        items.append(
            PlanItem(
                household=callback.household,
                checkpoint_id=checkpoint.id if checkpoint else "",
                episode=checkpoint.episode if checkpoint else "",
                due=callback.due,
                week=week,
                kind=KIND_CALLBACK,
                site=callback.site,
                channel=channel,
                probability=probability_by_checkpoint.get(checkpoint.id) if checkpoint else None,
                uses_check_slot=True,
                uses_change_slot=False,
                callback_id=callback.id,
                reason=(
                    "按已约定的回拨时间联系；优先于新机会占用查证名额"
                    + (f"；当天协议状态：{exposure.label}" if exposure.state != AT_RISK else "")
                ),
                next_week_check=(
                    _next_due_text(data, checkpoint) if checkpoint else "核对回拨后的安排"
                ),
            )
        )
        if checkpoint:
            handled.add(checkpoint.id)

    # 2) 其余本周到期的应约日按风险从高到低安排。
    candidates = [
        checkpoint
        for checkpoint in data.checkpoints
        if week <= checkpoint.due < week + timedelta(days=7)
        and checkpoint.id not in handled
    ]
    candidates.sort(
        key=lambda row: (
            -(probability_by_checkpoint.get(row.id) or 0.0),
            row.due,
            row.id,
        )
    )
    mapping = spell_ids(data, as_of)
    for checkpoint in candidates:
        household = data.household_of(checkpoint)
        permission = data.permission(household)
        probability = probability_by_checkpoint.get(checkpoint.id)
        exposure = exposure_at(data, checkpoint, as_of)
        if exposure.state != AT_RISK:
            items.append(
                _skip(
                    data, checkpoint, week, probability, None,
                    f"本周不安排原协议联系：{exposure.reason}",
                    "核对替代安排是否按实际期限结束",
                )
            )
            continue
        if permission is None:
            items.append(
                _skip(data, checkpoint, week, probability, None,
                      "没有联络授权记录", "补齐授权后再排期")
            )
            continue
        if not permission.active:
            items.append(
                _skip(data, checkpoint, week, probability, None,
                      "家庭已撤回联络授权，不得主动联系", "仅保留被动回单核查")
            )
            continue
        if permission.starts > checkpoint.due:
            items.append(
                _skip(data, checkpoint, week, probability, None,
                      f"联络授权从 {permission.starts.isoformat()} 才生效，本周不能主动联系",
                      "授权生效后再评估联系")
            )
            continue
        site, basis = authority_site(data, household, checkpoint.due, as_of)
        if site is None:
            items.append(
                _skip(data, checkpoint, week, probability, None,
                      f"没有可落实服务的站点：{basis}", "确认是否已迁走或转出")
            )
            continue
        if site != checkpoint_episode_site(data, checkpoint):
            items.append(
                _skip(
                    data, checkpoint, week, probability, site,
                    f"另一机构（{site}）正在完成同一家庭的转接，避免重复联系",
                    f"由 {site} 在交接完成后反馈结果",
                )
            )
            continue
        if site not in ledger:
            items.append(
                _skip(data, checkpoint, week, probability, site,
                      "本周没有该站点的余量资料，无法安排联系",
                      "补录余量后重新排期")
            )
            continue
        slots = ledger[site]
        arrange_ok = (
            probability is not None
            and probability >= ARRANGE_THRESHOLD
            and slots["change_total"] > 0
            and slots["change_used"] < slots["change_total"]
            and slots["check_used"] < slots["check_total"]
            and exposure.state == AT_RISK
        )
        if arrange_ok:
            slots["check_used"] += 1
            slots["change_used"] += 1
            items.append(
                PlanItem(
                    household=household,
                    checkpoint_id=checkpoint.id,
                    episode=checkpoint.episode,
                    due=checkpoint.due,
                    week=week,
                    kind=KIND_ARRANGE,
                    site=site,
                    channel=permission.channel,
                    probability=probability,
                    uses_check_slot=True,
                    uses_change_slot=True,
                    callback_id="",
                    reason=(
                        f"中断概率较高（{probability:.0%}），站点 {site} 对当前协议有权"
                        "且本周有可落实的改约额度"
                    ),
                    next_week_check=_next_due_text(data, checkpoint),
                )
            )
            continue
        if slots["check_used"] < slots["check_total"]:
            slots["check_used"] += 1
            reason = (
                "先查证：本周没有改约额度或改约条件不满足"
                if slots["change_total"] == 0 or slots["change_used"] >= slots["change_total"]
                else "先查证：预测概率未达到直接改约的阈值，先确认服务是否落实"
            )
            items.append(
                PlanItem(
                    household=household,
                    checkpoint_id=checkpoint.id,
                    episode=checkpoint.episode,
                    due=checkpoint.due,
                    week=week,
                    kind=KIND_DOCUMENT,
                    site=site,
                    channel=permission.channel,
                    probability=probability,
                    uses_check_slot=True,
                    uses_change_slot=False,
                    callback_id="",
                    reason=reason,
                    next_week_check=_next_due_text(data, checkpoint),
                )
            )
            continue
        items.append(
            _skip(data, checkpoint, week, probability, site,
                  "本周查证名额已满，未能安排联系", "下周优先补查，并记录本周未联系")
        )

    slots_view = tuple(
        SlotLedger(
            site=site,
            week=week,
            check_total=values["check_total"],
            check_used=values["check_used"],
            change_total=values["change_total"],
            change_used=values["change_used"],
        )
        for site, values in sorted(ledger.items())
    )
    if not ledger:
        notes.append("该周没有站点余量资料，所有联系都只能列为待排期。")
    notes.append(
        "计划只列本周可执行的联系；下周要回到实际回单与已确认安排，"
        "未确认的记录不当作缺口，也不当作有服务。"
    )
    if predictions and all(
        prediction.limitation and "训练样本不足" in prediction.limitation
        for prediction in predictions
    ):
        notes.append("预测处于低支持状态：训练样本不足，计划只宜作核查与记录用途。")
    if any(prediction.band == "high" for prediction in predictions) and predictions and (
        len([p for p in predictions if p.state == AT_RISK]) < 10
    ):
        notes.append("本周在险个案很少，排序差异主要来自特征外推，宜同时保留人工判断。")
    if any(item.kind == KIND_SKIP and "撤回" in item.reason for item in items):
        notes.append("存在明确撤回的联络授权；这些家庭既不联系，也不进入新的联系机会。")

    order = {KIND_CALLBACK: 0, KIND_ARRANGE: 1, KIND_DOCUMENT: 2, KIND_SKIP: 3}
    items.sort(key=lambda item: (order[item.kind], -(item.probability or 0.0), item.due, item.household))
    return WeeklyPlan(week=week, as_of=as_of, items=tuple(items), slots=slots_view, notes=tuple(notes))


def checkpoint_episode_site(data: PolicyData, checkpoint: Checkpoint) -> str:
    return data.episode(checkpoint.episode).site


def _skip(
    data: PolicyData,
    checkpoint: Checkpoint,
    week: date,
    probability: float | None,
    site: str | None,
    reason: str,
    next_week_check: str,
) -> PlanItem:
    return PlanItem(
        household=data.household_of(checkpoint),
        checkpoint_id=checkpoint.id,
        episode=checkpoint.episode,
        due=checkpoint.due,
        week=week,
        kind=KIND_SKIP,
        site=site,
        channel=(data.permission(data.household_of(checkpoint)).channel
                 if data.permission(data.household_of(checkpoint)) else "未登记"),
        probability=probability,
        uses_check_slot=False,
        uses_change_slot=False,
        callback_id="",
        reason=reason,
        next_week_check=next_week_check,
    )


# --------------------------------------------------------------------------
# 计划执行后的对账
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanReconciliation:
    """一条计划在下周的事实与历史复盘口径。"""

    household: str
    checkpoint_id: str
    planned_kind: str
    planned_reason: str
    next_week_outcome: str
    next_week_note: str
    historical_outcome: str
    historical_note: str


def reconcile_plan(
    data: PolicyData,
    plan: WeeklyPlan,
    next_as_of: date,
) -> tuple[PlanReconciliation, ...]:
    """把一周计划对回下周可用事实，同时单列历史复盘口径。

    * ``next_week_outcome``：在 ``next_as_of`` 能确认到什么程度（待回单、渠道未查齐、
      已证实有服务/无服务/由替代安排覆盖）；
    * ``historical_outcome``：资料最终可见的结果，用于历史复盘；
    * 两者不同时明确写出差异，避免把下周还没查清的事当成已确认缺口。

    没有应约记录的约定回拨按下周的事实记录核对；本函数不把回拨结果推断成服务缺口。
    """
    from .timeline import (
        classify_outcome,
        contamination_reason,
        final_outcome,
        outcome_label,
    )

    rows: list[PlanReconciliation] = []
    for item in plan.items:
        if not item.checkpoint_id:
            rows.append(
                PlanReconciliation(
                    household=item.household,
                    checkpoint_id="",
                    planned_kind=item.kind,
                    planned_reason=item.reason,
                    next_week_outcome="no_scheduled_visit",
                    next_week_note="没有对应应约记录；只核对联络记录本身，不推断服务缺口",
                    historical_outcome="no_scheduled_visit",
                    historical_note="历史复盘同样不把回拨当作服务结果",
                )
            )
            continue
        checkpoint = data.checkpoint(item.checkpoint_id)
        now = classify_outcome(checkpoint, next_as_of)
        final = final_outcome(checkpoint)
        contamination = contamination_reason(data, checkpoint)
        if item.kind in (KIND_DOCUMENT, KIND_ARRANGE):
            now_note = (
                f"计划为{'改约' if item.kind == KIND_ARRANGE else '查证'}；"
                f"下周只能确认到“{outcome_label(now)}”"
            )
            if contamination:
                now_note += "；这次联系改变了回单内容，不能与未联系的同经历记录混在一起复盘"
        else:
            now_note = f"未安排联系；下周确认到“{outcome_label(now)}”"
        final_note = f"历史复盘口径为“{outcome_label(final)}”"
        if now != final:
            final_note += f"（与下周口径不同：下周未确认完的仍按“{outcome_label(now)}”处理）"
        rows.append(
            PlanReconciliation(
                household=item.household,
                checkpoint_id=item.checkpoint_id,
                planned_kind=item.kind,
                planned_reason=item.reason,
                next_week_outcome=now,
                next_week_note=now_note,
                historical_outcome=final,
                historical_note=final_note,
            )
        )
    return tuple(rows)
