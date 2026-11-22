"""在某个工作日做中断预测。

预测使用“那时能得到的记录”：

* 应约日尚未到的记录不可见；
* 回单尚未收到的应约日按“待回单”处理，不能当作有服务或无服务；
* 变更只在 ``recorded_on <= as_of`` 时可见；
* 同一家庭更早协议的经历可以参照，但同样要等到其回单可见；
* 同一风险经历内更早的应约日可以进入特征，但不能把结局提前。

模型是一个带脊罚的 logistic 回归，只用标准库。它不追求复杂，而是把数据里
能稳定看到的信息（最近一次已确认结果、连续有服务的次数、先前缺口次数、先前未
查齐次数、协议长度与站点）纳入，并在缺少足够训练样本时退回整体缺口率。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math

from .data import Checkpoint, PolicyData
from .timeline import (
    AT_RISK,
    COVERED,
    COVERED_ELSEWHERE,
    GAP,
    SERVED,
    UNCONFIRMED,
    classify_outcome,
    exposure_at,
    gap_label,
    outreach_contaminated,
    spell_ids,
)

# 特征名称；顺序固定，便于解释与测试。
CATEGORICAL = {
    "last": ("none", "served", "gap", "unconfirmed", "covered"),
    "group": ("regular", "remote", "support"),
    "site": ("S1", "S2", "S3", "S4"),
}
NUMERIC = ("served_streak", "prior_gaps", "prior_unconfirmed", "seen_count", "cross_episode")

DEFAULT_RIDGE = 2.0
DEFAULT_ITERATIONS = 400
DEFAULT_RATE = 0.15
MIN_TRAIN_ROWS = 40


@dataclass(frozen=True)
class RiskFeatures:
    checkpoint_id: str
    household: str
    episode: str
    group: str
    site: str
    due: date
    last_state: str
    served_streak: int
    prior_gaps: int
    prior_unconfirmed: int
    seen_count: int
    cross_episode: int
    observation_days: int
    spell: str

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_id,
            "household": self.household,
            "episode": self.episode,
            "group": self.group,
            "site": self.site,
            "due": self.due.isoformat(),
            "last_state": self.last_state,
            "served_streak": self.served_streak,
            "prior_gaps": self.prior_gaps,
            "prior_unconfirmed": self.prior_unconfirmed,
            "seen_count": self.seen_count,
            "cross_episode": self.cross_episode,
            "observation_days": self.observation_days,
            "spell": self.spell,
        }

    def vector(self) -> list[float]:
        values: list[float] = []
        for name, categories in CATEGORICAL.items():
            observed = {
                "last": self.last_state,
                "group": self.group,
                "site": self.site,
            }[name]
            for category in categories[1:]:
                values.append(1.0 if observed == category else 0.0)
        values.extend(
            float(item)
            for item in (
                self.served_streak,
                self.prior_gaps,
                self.prior_unconfirmed,
                self.seen_count,
                self.cross_episode,
            )
        )
        return values


def _ordered_household_checkpoints(data: PolicyData, household: str) -> tuple[Checkpoint, ...]:
    return data.checkpoints_for_household(household)


def features_for(
    data: PolicyData,
    checkpoint: Checkpoint,
    as_of: date,
    *,
    spell_map: dict[str, str] | None = None,
) -> RiskFeatures | None:
    """构造某个应约日在 ``as_of`` 的特征；该应约日不在风险中时返回 ``None``。

    特征只统计 ``as_of`` 时已经收到回单、且当时判断处于风险中的更早应约日。
    同家庭更早协议的经历单独用 ``cross_episode`` 标记，不与本协议混淆。
    """
    if exposure_at(data, checkpoint, as_of).state != AT_RISK:
        return None
    episode = data.episode(checkpoint.episode)
    household_rows = _ordered_household_checkpoints(data, episode.household)
    prior = [
        row
        for row in household_rows
        if row.due < checkpoint.due
        and row.received <= as_of
        and exposure_at(data, row, as_of).state in (AT_RISK, COVERED)
    ]
    at_risk = [
        row for row in prior if exposure_at(data, row, as_of).state == AT_RISK
    ]
    same_episode = [row for row in at_risk if row.episode == checkpoint.episode]
    all_known = sorted(at_risk, key=lambda row: (row.due, row.id))
    latest = sorted(prior, key=lambda row: (row.due, row.id))

    def state(row: Checkpoint) -> str:
        outcome = classify_outcome(row, as_of)
        return {
            SERVED: "served",
            GAP: "gap",
            UNCONFIRMED: "unconfirmed",
            COVERED_ELSEWHERE: "covered",
        }.get(outcome, "unconfirmed")

    last_state = state(latest[-1]) if latest else "none"
    streak = 0
    for row in reversed(all_known):
        if state(row) == "served":
            streak += 1
        else:
            break
    prior_gaps = sum(1 for row in same_episode if gap_label(classify_outcome(row, as_of)) == GAP)
    prior_unconfirmed = sum(
        1 for row in same_episode if classify_outcome(row, as_of) == UNCONFIRMED
    )
    cross_episode = 1 if any(row.episode != checkpoint.episode for row in prior) else 0
    spell_map = spell_map if spell_map is not None else spell_ids(data, as_of)
    return RiskFeatures(
        checkpoint_id=checkpoint.id,
        household=episode.household,
        episode=episode.episode,
        group=episode.group,
        site=episode.site,
        due=checkpoint.due,
        last_state=last_state,
        served_streak=min(streak, 3),
        prior_gaps=min(prior_gaps, 2),
        prior_unconfirmed=min(prior_unconfirmed, 2),
        seen_count=min(len(same_episode), 3),
        cross_episode=cross_episode,
        observation_days=(checkpoint.due - episode.begins).days,
        spell=spell_map.get(checkpoint.id, ""),
    )


@dataclass(frozen=True)
class TrainingRow:
    features: RiskFeatures
    label: str  # "gap" 或 "no_gap"
    checkpoint_id: str
    spell: str


def training_rows(
    data: PolicyData,
    as_of: date,
    *,
    exclude_spells: frozenset[str] = frozenset(),
    spell_map: dict[str, str] | None = None,
) -> tuple[TrainingRow, ...]:
    """``as_of`` 时已经可以用于训练的历史记录。

    * 只纳入 ``as_of`` 时已收到回单的应约日；
    * 只纳入 ``as_of`` 时仍处于风险中的应约日（暂停、替代、终止后的不承担中断风险）；
    * “渠道尚未查齐”不进入二分标签，但会作为后续记录的特征；
    * 执行日早于回单日的联络会改变后来查到的结果，因此排除出训练，避免把
      联系得最勤的个案当成预警效果；
    * ``exclude_spells`` 用于留出同一经历，避免前后混入训练与复盘。
    """
    mapping = spell_map if spell_map is not None else spell_ids(data, as_of)
    rows: list[TrainingRow] = []
    for checkpoint in data.checkpoints:
        if checkpoint.received > as_of or checkpoint.due > as_of:
            continue
        if exposure_at(data, checkpoint, as_of).state != AT_RISK:
            continue
        # 训练只用 ``as_of`` 时的判断：此后才登记的迁走或替代不能提前进入训练。
        outcome = classify_outcome(checkpoint, as_of)
        label = gap_label(outcome)
        if label not in ("gap", "no_gap"):
            # 渠道未查齐的结局不进入二分标签，但会作为后续记录的特征。
            continue
        spell = mapping.get(checkpoint.id, "")
        if spell in exclude_spells:
            continue
        if not spell:
            continue
        if outreach_contaminated(data, checkpoint):
            continue
        features = features_for(data, checkpoint, as_of, spell_map=mapping)
        if features is None:
            continue
        rows.append(
            TrainingRow(
                features=features,
                label="gap" if outcome == GAP else "no_gap",
                checkpoint_id=checkpoint.id,
                spell=spell,
            )
        )
    return tuple(rows)


def _label_targets(rows: tuple[TrainingRow, ...]) -> tuple[list[list[float]], list[int]]:
    return [row.features.vector() for row in rows], [1 if row.label == "gap" else 0 for row in rows]


def fit_logistic(
    rows: tuple[TrainingRow, ...],
    *,
    ridge: float = DEFAULT_RIDGE,
    iterations: int = DEFAULT_ITERATIONS,
    rate: float = DEFAULT_RATE,
) -> "RiskModel":
    """用全批量梯度下降拟合；样本不足时返回只含基础概率的模型。"""
    if len(rows) < MIN_TRAIN_ROWS:
        return RiskModel(base_rate=_base_rate(rows), weights=(), intercept=0.0, rows=len(rows))
    vectors, targets = _label_targets(rows)
    n = len(vectors)
    width = len(vectors[0])
    weights = [0.0] * width
    positives = sum(targets)
    intercept = math.log(max(positives, 1) / max(n - positives, 1))
    exp = math.exp
    # 全批量梯度下降；内层循环用局部变量，避免在热路径上反复建生成器。
    for _ in range(iterations):
        gradient = [0.0] * width
        gradient_b = 0.0
        for vector, target in zip(vectors, targets):
            score = intercept
            for index in range(width):
                score += weights[index] * vector[index]
            predicted = 1.0 / (1.0 + exp(-score))
            error = predicted - target
            gradient_b += error
            for index in range(width):
                gradient[index] += error * vector[index]
        for index in range(width):
            weights[index] -= rate * (gradient[index] / n + ridge * weights[index] / n)
        intercept -= rate * gradient_b / n
    return RiskModel(
        base_rate=_base_rate(rows),
        weights=tuple(weights),
        intercept=intercept,
        rows=n,
    )


def _base_rate(rows: tuple[TrainingRow, ...]) -> float:
    if not rows:
        return 0.10
    gaps = sum(1 for row in rows if row.label == "gap")
    return (gaps + 1.0) / (len(rows) + 2.0)


@dataclass(frozen=True)
class RiskModel:
    """已拟合的中断风险模型。"""

    base_rate: float
    weights: tuple[float, ...]
    intercept: float
    rows: int

    @property
    def fallback(self) -> bool:
        return not self.weights

    def predict(self, features: RiskFeatures) -> float:
        if self.fallback:
            return self.base_rate
        score = self.intercept + sum(
            weight * value for weight, value in zip(self.weights, features.vector())
        )
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))


@dataclass(frozen=True)
class Prediction:
    checkpoint_id: str
    household: str
    episode: str
    due: date
    probability: float
    state: str = AT_RISK
    band: str = ""
    limitation: str = ""
    model_rows: int = 0
    features: RiskFeatures | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_id,
            "household": self.household,
            "episode": self.episode,
            "due": self.due.isoformat(),
            "probability": round(self.probability, 4),
            "state": self.state,
            "band": self.band,
            "limitation": self.limitation,
            "model_rows": self.model_rows,
        }


def band_for(probability: float) -> str:
    if probability >= 0.30:
        return "high"
    if probability >= 0.15:
        return "moderate"
    return "low"


def predict_week(
    data: PolicyData,
    as_of: date,
    *,
    horizon_days: int = 7,
    model: RiskModel | None = None,
) -> tuple[Prediction, ...]:
    """预测 ``as_of`` 之后 ``horizon_days`` 天内到期的应约日。

    只用 ``as_of`` 时已有的记录训练和构造特征。尚未到期的应约日全部保留，
    包括未来才到约定日期的记录（概率按当时可见信息给出，并标注限制）。
    """
    if model is None:
        model = fit_logistic(training_rows(data, as_of))
    mapping = spell_ids(data, as_of)
    window_end = as_of + timedelta(days=horizon_days)
    predictions: list[Prediction] = []
    for checkpoint in data.checkpoints:
        if not (as_of <= checkpoint.due < window_end):
            continue
        exposure = exposure_at(data, checkpoint, as_of)
        if exposure.state != AT_RISK:
            predictions.append(
                Prediction(
                    checkpoint_id=checkpoint.id,
                    household=data.household_of(checkpoint),
                    episode=checkpoint.episode,
                    due=checkpoint.due,
                    probability=0.0,
                    state=exposure.state,
                    band="not_at_risk",
                    limitation=exposure.reason,
                    model_rows=model.rows,
                )
            )
            continue
        features = features_for(data, checkpoint, as_of, spell_map=mapping)
        if features is None:
            continue
        probability = model.predict(features)
        limitation = ""
        if model.fallback:
            limitation = "训练样本不足，暂用整体缺口率；只能作描述性参考"
        elif not features.seen_count:
            limitation = "该协议尚无已确认回单，预测主要依赖同家庭更早协议与基础概率"
        elif features.observation_days < 14:
            limitation = "协议刚开始，观察窗偏短，概率不确定性较大"
        predictions.append(
            Prediction(
                checkpoint_id=checkpoint.id,
                household=features.household,
                episode=features.episode,
                due=features.due,
                probability=probability,
                band=band_for(probability),
                limitation=limitation,
                model_rows=model.rows,
                features=features,
            )
        )
    predictions.sort(key=lambda row: (-row.probability, row.due, row.checkpoint_id))
    return tuple(predictions)


def explain(features: RiskFeatures) -> tuple[str, ...]:
    """给出可读的特征说明，供主管按个案查看。"""
    reasons: list[str] = []
    if features.last_state == "none":
        reasons.append("本协议还没有可用的回单记录")
    elif features.last_state == "served":
        reasons.append("最近一次已确认有服务")
    elif features.last_state == "gap":
        reasons.append("最近一次已确认无服务")
    elif features.last_state == "unconfirmed":
        reasons.append("最近一次渠道尚未查齐")
    else:
        reasons.append("最近一次由既有替代安排覆盖")
    if features.served_streak:
        reasons.append(f"连续 {features.served_streak} 次已确认有服务")
    if features.prior_gaps:
        reasons.append(f"本协议内已有 {features.prior_gaps} 次确认缺口")
    if features.prior_unconfirmed:
        reasons.append(f"本协议内已有 {features.prior_unconfirmed} 次渠道未查齐")
    if features.cross_episode:
        reasons.append("参照了同家庭更早协议的已确认回单")
    return tuple(reasons)
