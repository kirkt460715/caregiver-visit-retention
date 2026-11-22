"""协议期限、已确认变更、应约与回单的时点口径。

本模块只回答三类问题：

* 某个应约日在预测时点看来处于什么状态（仍受原协议约束、暂停、由替代安排覆盖、
  还是原协议已经终止）；
* 某个应约日的结果在观察时点能确认到什么程度（未到期、待回单、渠道未查齐、
  已证实有服务、已证实无服务、由既有替代安排覆盖）；
* 同一家庭的风险经历（spell）如何划分，供训练、复盘分组使用。

所有判断都接受 ``as_of``。只有 ``recorded_on <= as_of`` 的变更才算已知；已知变更
按 ``starts``/``ends`` 的实际期限生效，左闭右开。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .data import Checkpoint, Episode, PolicyData, ServiceChange

# 暴露状态
AT_RISK = "at_risk"  # 仍受原协议约束，存在中断风险
SUSPENDED = "suspended"  # 已确认的暂停/暂住，期间不承担原协议中断风险
COVERED = "covered"  # 已确认的机构替代/既有替代安排覆盖
ENDED = "ended"  # 家庭迁走或转出，原协议终止，此后不再有原协议中断风险

# 结果状态
NOT_DUE = "not_due"  # 应约日尚未到
AWAITING = "awaiting_receipt"  # 已到期但回单尚未收到
UNCONFIRMED = "unconfirmed"  # 回单已到但渠道未查齐
SERVED = "served"  # 有服务
COVERED_ELSEWHERE = "covered_elsewhere"  # 无原服务，但由既有替代安排覆盖
GAP = "gap"  # 各渠道都证实无服务，且没有替代安排

RESOLVED_OUTCOMES = frozenset({SERVED, COVERED_ELSEWHERE, GAP})
# 二分口径：只有“各渠道证实无服务”算服务缺口。
OUTCOME_GAP = "gap"
OUTCOME_NO_GAP = "no_gap"


@dataclass(frozen=True)
class Exposure:
    """一个应约日在某个时点的协议状态。"""

    state: str
    reason: str
    change: ServiceChange | None = None

    @property
    def at_risk(self) -> bool:
        return self.state == AT_RISK

    @property
    def label(self) -> str:
        return {
            AT_RISK: "仍受原协议约束",
            SUSPENDED: "已确认暂停/暂住",
            COVERED: "已由替代安排覆盖",
            ENDED: "原协议已终止",
        }[self.state]


_CHANGE_PRIORITY = {
    # 终止类优先于覆盖类，再优先于暂停类。
    "moved": 0,
    "transferred": 0,
    "new_alternate": 1,
    "temporary_stay": 2,
}


def _effective_change(
    changes: tuple[ServiceChange, ...], day: date
) -> ServiceChange | None:
    active = [change for change in changes if change.covers(day)]
    if not active:
        return None
    return min(
        active,
        key=lambda change: (_CHANGE_PRIORITY[change.status], change.starts, change.id),
    )


def _exposure_for_change(change: ServiceChange | None) -> Exposure:
    if change is None:
        return Exposure(AT_RISK, "没有已知的暂停、替代或终止安排")
    if change.status == "moved":
        return Exposure(ENDED, f"家庭迁走（{change.starts.isoformat()} 起）", change)
    if change.status == "transferred":
        return Exposure(
            ENDED,
            f"已转出至 {change.new_provider}（{change.starts.isoformat()} 起）",
            change,
        )
    if change.status == "new_alternate":
        target = change.new_provider or "其他机构"
        until = change.ends.isoformat() if change.ends else "持续"
        return Exposure(
            COVERED,
            f"已确认由 {target} 提供替代服务"
            f"（{change.starts.isoformat()} 至 {until}）",
            change,
        )
    if change.status == "temporary_stay":
        until = change.ends.isoformat() if change.ends else "持续"
        return Exposure(
            SUSPENDED,
            f"与家人同住/暂住（{change.starts.isoformat()} 至 {until}）",
            change,
        )
    return Exposure(AT_RISK, "未知变更状态")


def exposure_for_episode(
    data: PolicyData, episode: Episode, day: date, as_of: date
) -> Exposure:
    """按协议判断某日的状态；只使用 ``as_of`` 时已经登记的变更。"""
    return _exposure_for_change(
        _effective_change(data.known_changes(episode.episode, as_of), day)
    )


def exposure_at(data: PolicyData, checkpoint: Checkpoint, as_of: date) -> Exposure:
    """从原协议是否仍对该家庭有约束的角度，判断应约日的状态。

    ``as_of`` 之前未登记的变更不可见。已登记的变更按实际起止日生效，而不是从
    登记日生效。家庭迁走或转出后，原协议终止，之后不再计算原协议的中断风险。
    """
    return exposure_for_episode(data, data.episode(checkpoint.episode), checkpoint.due, as_of)


def exposure_for_household(
    data: PolicyData, household: str, day: date, as_of: date
) -> Exposure:
    """按家庭判断某日的协议状态；用于没有应约记录的约定回拨。"""
    episodes = data.episodes_by_household.get(household, ())
    covering = [episode for episode in episodes if episode.begins <= day < episode.ends]
    if covering:
        return exposure_for_episode(data, covering[0], day, as_of)
    changes = tuple(
        change
        for episode in episodes
        for change in data.known_changes(episode.episode, as_of)
        if change.covers(day)
    )
    change = _effective_change(changes, day)
    if change is not None:
        return _exposure_for_change(change)
    if episodes and day >= max(episode.ends for episode in episodes):
        return Exposure(ENDED, "现有协议已结束，且没有已知的承接安排")
    return Exposure(AT_RISK, "没有已知的暂停、替代或终止安排")


def authority_site(
    data: PolicyData, household: str, day: date, as_of: date
) -> tuple[str | None, str]:
    """返回某日对当前安排有处理权的站点及依据。

    * 已确认迁走：没有可联系的原协议站点；
    * 已转出或由其他机构替代：承接机构优先，原站点不得重复联系；
    * 暂停/暂住：原站点仍是有权站点，但期间不承担中断风险；
    * 其他：当前协议站点。
    """
    episodes = data.episodes_by_household.get(household, ())
    covering = [episode for episode in episodes if episode.begins <= day < episode.ends]
    changes = tuple(
        change
        for episode in episodes
        for change in data.known_changes(episode.episode, as_of)
        if change.covers(day)
    )
    change = _effective_change(changes, day)
    if change is not None:
        if change.status == "moved":
            return None, "家庭迁走，原协议已终止"
        if change.status in {"transferred", "new_alternate"} and change.new_provider:
            return change.new_provider, "承接机构优先"
        if change.status == "temporary_stay":
            site = covering[0].site if covering else (episodes[-1].site if episodes else None)
            return site, "暂停/暂住期间原站点仍有权，但无中断风险"
    if covering:
        return covering[0].site, "当前协议站点"
    if episodes and day >= max(episode.ends for episode in episodes):
        return None, "现有协议已结束，且没有已知的承接安排"
    return (episodes[-1].site if episodes else None), "没有覆盖该日的协议"


def classify_outcome(checkpoint: Checkpoint, as_of: date) -> str:
    """在 ``as_of`` 时点对该应约日结果能确认到什么程度。"""
    if checkpoint.due > as_of:
        return NOT_DUE
    if checkpoint.received > as_of:
        return AWAITING
    if checkpoint.receipt_state == "unresolved":
        return UNCONFIRMED
    if checkpoint.receipt_state == "verified_yes":
        return SERVED
    if checkpoint.receipt_state == "verified_no":
        if checkpoint.context == "alternate_preexisting":
            return COVERED_ELSEWHERE
        if checkpoint.context == "normal":
            return GAP
        return UNCONFIRMED
    return UNCONFIRMED


def final_outcome(checkpoint: Checkpoint) -> str:
    return classify_outcome(checkpoint, date.max)


def outcome_label(state: str) -> str:
    return {
        NOT_DUE: "尚未到期",
        AWAITING: "已到期、待回单",
        UNCONFIRMED: "渠道尚未查齐",
        SERVED: "已证实有服务",
        COVERED_ELSEWHERE: "由既有替代安排覆盖",
        GAP: "各渠道证实无服务",
    }[state]


def gap_label(state: str) -> str:
    """把结果压成评估用的二分口径；未确认的结局不进入二分标签。"""
    if state == GAP:
        return OUTCOME_GAP
    if state in {SERVED, COVERED_ELSEWHERE}:
        return OUTCOME_NO_GAP
    return state


def spell_ids(data: PolicyData, as_of: date | None = None) -> dict[str, str]:
    """为每个应约日分配风险经历编号；不处于风险中的应约日编号为空串。

    风险经历是一段连续的、仍受原协议约束的服务期。以下情况开始新的一段：

    * 应约日不处于风险中（暂停、替代覆盖、终止）；
    * 换了一段协议（重新开通）；
    * 应约间隔超过一周（中间没有约定服务）。

    评估时同一经历的前后记录应整段进入训练或整段进入复盘。
    """
    as_of = as_of or date.max
    result: dict[str, str] = {}
    for household, episodes in data.episodes_by_household.items():
        ordered: list[Checkpoint] = []
        for episode in episodes:
            ordered.extend(data.checkpoints_by_episode.get(episode.episode, ()))
        ordered.sort(key=lambda row: (row.due, row.id))
        previous: Checkpoint | None = None
        previous_at_risk = False
        current: str | None = None
        counters: dict[str, int] = {}
        for checkpoint in ordered:
            at_risk = exposure_at(data, checkpoint, as_of).at_risk
            continuous = (
                at_risk
                and previous_at_risk
                and previous is not None
                and previous.episode == checkpoint.episode
                and checkpoint.due - previous.due <= timedelta(days=7)
            )
            if not at_risk:
                result[checkpoint.id] = ""
                current = None
            else:
                if not continuous:
                    counters[checkpoint.episode] = counters.get(checkpoint.episode, 0) + 1
                    current = f"{household}:{checkpoint.episode}:{counters[checkpoint.episode]}"
                result[checkpoint.id] = current or ""
            previous = checkpoint
            previous_at_risk = at_risk
    return result


@dataclass(frozen=True)
class Spell:
    id: str
    household: str
    episode: str
    first_due: date
    last_due: date
    checkpoint_ids: tuple[str, ...]



def spells(data: PolicyData, as_of: date | None = None) -> dict[str, Spell]:
    mapping = spell_ids(data, as_of)
    grouped: dict[str, list[Checkpoint]] = {}
    for checkpoint in data.checkpoints:
        spell = mapping.get(checkpoint.id)
        if spell:
            grouped.setdefault(spell, []).append(checkpoint)
    result: dict[str, Spell] = {}
    for spell, rows in grouped.items():
        rows.sort(key=lambda row: row.due)
        result[spell] = Spell(
            id=spell,
            household=rows[0].episode.split("-")[0],
            episode=rows[0].episode,
            first_due=rows[0].due,
            last_due=rows[-1].due,
            checkpoint_ids=tuple(row.id for row in rows),
        )
    return result


def outreach_contaminated(data: PolicyData, checkpoint: Checkpoint) -> bool:
    """联系本身是否可能改变后来查到的结果。

    执行日早于回单日时，这次联系可能改变回单内容，该应约日不能作为评价联络
    效果的干净对照。没有执行日的记录不受影响。
    """
    outreach = data.outreach_by_checkpoint.get(checkpoint.id)
    if outreach is None or outreach.action_day is None:
        return False
    return outreach.action_day < checkpoint.received


def contamination_reason(data: PolicyData, checkpoint: Checkpoint) -> str:
    outreach = data.outreach_by_checkpoint.get(checkpoint.id)
    if outreach is None or outreach.action_day is None:
        return ""
    if outreach.action_day < checkpoint.received:
        return (
            f"执行日 {outreach.action_day.isoformat()} 早于回单日 "
            f"{checkpoint.received.isoformat()}，联系可能改变后来查到的结果"
        )
    return ""
