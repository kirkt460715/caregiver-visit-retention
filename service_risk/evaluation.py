"""预警与联络策略的评估。

评估分三件事，口径分开：

1. **预警能否找准服务缺口**：按周前向回测。每次到达某个工作日只用当时能得到的
   记录训练，再用之后一周内到期的应约日复盘；同一风险经历整段留出，不把前后
   混入训练与复盘。
2. **联络策略能否改变结果**：使用已登记的分派概率做意向性比较（ITT）。观察窗
   从应约日开始，到下一次约定服务为止；执行日早于回单日的记录会改变后来查到
   的结果，单独标出，不混入效果估计。
3. **哪些记录只能描述**：非登记分派、样本过小、观察窗未成熟的记录只做描述。

“主动联系后收到另一机构已接手的证明”改变的是结果能否确认（由无服务变为有替代
安排覆盖）；“联络后新安排了人”才可能真的避免中断。两者在结果字段与观察窗里
分别处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import random

from .data import Checkpoint, Outreach, PolicyData
from .risk import (
    MIN_TRAIN_ROWS,
    Prediction,
    RiskFeatures,
    features_for,
    fit_logistic,
    training_rows,
)
from .timeline import (
    AT_RISK,
    classify_outcome,
    contamination_reason,
    exposure_at,
    final_outcome,
    gap_label,
    outcome_label,
    spell_ids,
)

SEED = 20261122


# --------------------------------------------------------------------------
# 基础统计
# --------------------------------------------------------------------------
def _auc(scores: list[float], labels: list[int]) -> float | None:
    """带并列处理的 ROC AUC（Mann–Whitney U）。"""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _brier(scores: list[float], labels: list[int]) -> float | None:
    if not labels:
        return None
    return sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """二项比例的 Wilson 区间；样本为 0 时返回 ``None``。"""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


@dataclass(frozen=True)
class FoldMetrics:
    cutoff: date
    train_rows: int
    test_rows: int
    events: int
    auc: float | None
    brier: float | None
    baseline_brier: float | None
    base_rate: float
    model_rows: int = 0


@dataclass(frozen=True)
class Backtest:
    folds: tuple[FoldMetrics, ...]
    predictions: tuple[dict[str, object], ...]

    @property
    def test_rows(self) -> int:
        return sum(fold.test_rows for fold in self.folds)

    @property
    def events(self) -> int:
        return sum(fold.events for fold in self.folds)

    @property
    def auc(self) -> float | None:
        return _auc(
            [float(row["probability"]) for row in self.predictions],
            [int(row["label"]) for row in self.predictions],
        )

    @property
    def brier(self) -> float | None:
        return _brier(
            [float(row["probability"]) for row in self.predictions],
            [int(row["label"]) for row in self.predictions],
        )

    @property
    def baseline_brier(self) -> float | None:
        return _brier(
            [float(row["base_rate"]) for row in self.predictions],
            [int(row["label"]) for row in self.predictions],
        )

    @property
    def low_support(self) -> bool:
        """任一期训练样本不足，或复盘事件太少时，结论只宜方向性参考。"""
        return any(fold.model_rows < MIN_TRAIN_ROWS for fold in self.folds) or self.events < 20

    @property
    def calibration(self) -> tuple[CalibrationBin, ...]:
        bins: list[list[tuple[float, int]]] = [[] for _ in range(5)]
        for row in self.predictions:
            probability = float(row["probability"])
            index = min(int(probability / 0.2), 4)
            bins[index].append((probability, int(row["label"])))
        result: list[CalibrationBin] = []
        for index, items in enumerate(bins):
            if not items:
                continue
            result.append(
                CalibrationBin(
                    lower=index * 0.2,
                    upper=(index + 1) * 0.2,
                    count=len(items),
                    mean_predicted=sum(p for p, _ in items) / len(items),
                    observed_rate=sum(y for _, y in items) / len(items),
                )
            )
        return tuple(result)

    def summary(self) -> dict[str, object]:
        return {
            "folds": len(self.folds),
            "test_rows": self.test_rows,
            "events": self.events,
            "auc": self.auc,
            "brier": self.brier,
            "baseline_brier": self.baseline_brier,
            "event_rate": (self.events / self.test_rows) if self.test_rows else None,
            "low_support": self.low_support,
        }


def backtest(
    data: PolicyData,
    start: date,
    end: date,
    *,
    horizon_days: int = 7,
    step_days: int = 7,
) -> Backtest:
    """按工作日滚动前向回测。

    * 训练只用 ``cutoff`` 时已收到回单、且不在本周复盘经历内的记录；
    * 复盘记录是 ``[cutoff, cutoff + horizon)`` 内到期的应约日；
    * 标签用最终回单（复盘可以事后确认），特征只用 ``cutoff`` 时可见的事实；
    * 执行日早于回单日的联络记录不进入复盘，避免把联系本身的效果算成预警。

    ``end`` 是复盘窗口的最晚开始日；最后一个窗口的标签仍可来自更晚的回单。
    """
    folds: list[FoldMetrics] = []
    predictions: list[dict[str, object]] = []
    cutoff = start
    while cutoff < end:
        window_end = cutoff + timedelta(days=horizon_days)
        mapping = spell_ids(data, cutoff)
        test_candidates = [
            checkpoint
            for checkpoint in data.checkpoints
            if cutoff <= checkpoint.due < window_end
            and exposure_at(data, checkpoint, cutoff).state == AT_RISK
            and not contamination_reason(data, checkpoint)
        ]
        # 复盘标签用最终可见的协议状态：事后确认迁走、暂停或替代覆盖的，
        # 不是原协议下的缺口，不进入复盘。
        test_spells = frozenset(
            mapping.get(checkpoint.id, "") for checkpoint in test_candidates
        )
        train = training_rows(
            data,
            cutoff,
            exclude_spells=frozenset(spell for spell in test_spells if spell),
            spell_map=mapping,
        )
        model = fit_logistic(train)
        base_rate = model.base_rate
        rows: list[tuple[Prediction, RiskFeatures, int]] = []
        for checkpoint in test_candidates:
            if exposure_at(data, checkpoint, date.max).state != AT_RISK:
                continue
            final = classify_outcome(checkpoint, date.max)
            label = gap_label(final)
            if label not in ("gap", "no_gap"):
                continue
            features = features_for(data, checkpoint, cutoff, spell_map=mapping)
            if features is None:
                continue
            rows.append((Prediction(
                checkpoint_id=checkpoint.id,
                household=features.household,
                episode=features.episode,
                due=features.due,
                probability=model.predict(features),
                model_rows=model.rows,
                features=features,
            ), features, 1 if label == "gap" else 0))
        if len(train) >= MIN_TRAIN_ROWS and rows:
            scores = [prediction.probability for prediction, _, _ in rows]
            labels = [label for _, _, label in rows]
            folds.append(
                FoldMetrics(
                    cutoff=cutoff,
                    train_rows=len(train),
                    test_rows=len(rows),
                    events=sum(labels),
                    auc=_auc(scores, labels),
                    brier=_brier(scores, labels),
                    baseline_brier=_brier([base_rate] * len(labels), labels),
                    base_rate=base_rate,
                    model_rows=model.rows,
                )
            )
            for (prediction, features, label) in rows:
                predictions.append(
                    {
                        "cutoff": cutoff.isoformat(),
                        "checkpoint": prediction.checkpoint_id,
                        "episode": prediction.episode,
                        "household": prediction.household,
                        "due": prediction.due.isoformat(),
                        "probability": prediction.probability,
                        "label": label,
                        "base_rate": base_rate,
                        "spell": features.spell,
                    }
                )
        cutoff += timedelta(days=step_days)
    return Backtest(folds=tuple(folds), predictions=tuple(predictions))


# --------------------------------------------------------------------------
# 每个时点的暴露与结果分布
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExposureRow:
    checkpoint_id: str
    household: str
    due: date
    exposure: str
    exposure_reason: str
    outcome_now: str
    outcome_final: str
    note: str


def exposure_report(
    data: PolicyData, as_of: date, *, horizon_days: int = 7
) -> tuple[ExposureRow, ...]:
    """列出窗口内应约日在 ``as_of`` 的暴露状态与结果确认状态。

    “已到期且各渠道都证实无服务”“各渠道尚未查齐”“未来还没到约定日期”在这里
    分别呈现，不合并成一个“看不见回单”的类别。
    """
    window_end = as_of + timedelta(days=horizon_days)
    rows: list[ExposureRow] = []
    for checkpoint in data.checkpoints:
        if not (as_of <= checkpoint.due < window_end):
            continue
        exposure = exposure_at(data, checkpoint, as_of)
        now = classify_outcome(checkpoint, as_of)
        final = final_outcome(checkpoint)
        note = ""
        if exposure.state != AT_RISK:
            note = exposure.reason
        elif now != final:
            note = f"按 {as_of.isoformat()} 只能确认到“{outcome_label(now)}”"
        rows.append(
            ExposureRow(
                checkpoint_id=checkpoint.id,
                household=data.household_of(checkpoint),
                due=checkpoint.due,
                exposure=exposure.state,
                exposure_reason=exposure.reason,
                outcome_now=now,
                outcome_final=final,
                note=note,
            )
        )
    rows.sort(key=lambda row: (row.due, row.checkpoint_id))
    return tuple(rows)


# --------------------------------------------------------------------------
# 联络策略：意向性比较与证据强度
# --------------------------------------------------------------------------
def next_checkpoint(data: PolicyData, checkpoint: Checkpoint) -> Checkpoint | None:
    rows = data.checkpoints_by_episode.get(checkpoint.episode, ())
    for index, row in enumerate(rows):
        if row.id == checkpoint.id:
            return rows[index + 1] if index + 1 < len(rows) else None
    return None


OBSERVATION_MATURED = "观察窗已成熟"
OBSERVATION_PENDING = "观察窗未成熟"
OBSERVATION_CLOSED = "观察窗内协议已终止或无后续约定"


@dataclass(frozen=True)
class AssignmentEvidence:
    """一条联络分派在评估中的证据身份。"""

    outreach_id: str
    checkpoint_id: str
    household: str
    design: str
    arm: str
    target_outcome: str
    next_checkpoint_id: str
    next_outcome: str
    observation: str
    contamination: str
    identification: str
    reason: str

    @property
    def usable_for_effect(self) -> bool:
        """能否进入二分的效果估计：随机分派、观察窗成熟、且下一次结果已确认。"""
        return (
            self.identification == "randomized_itt"
            and self.observation == OBSERVATION_MATURED
            and gap_label(self.next_outcome) in ("gap", "no_gap")
        )


def assignment_evidence(data: PolicyData, as_of: date) -> tuple[AssignmentEvidence, ...]:
    """逐条说明联络分派能支持什么结论。

    识别强度分三档：

    * ``randomized_itt``：登记分派、概率已知，可在观察窗内做意向性比较；
    * ``descriptive_only``：非登记分派，只能描述已联络的个案；
    * ``not_usable``：执行早于回单（结果被联系改变）、观察窗未成熟/已关闭，
      或在 ``as_of`` 看来目标应约日本身已暂停、被覆盖或终止。
    """
    rows: list[AssignmentEvidence] = []
    for outreach in data.outreach:
        checkpoint = data.checkpoint(outreach.checkpoint)
        target = classify_outcome(checkpoint, date.max)
        following = next_checkpoint(data, checkpoint)
        contamination = contamination_reason(data, checkpoint)
        next_outcome = ""
        observation = OBSERVATION_PENDING
        if following is None:
            observation = OBSERVATION_CLOSED
        else:
            exposure = exposure_at(data, following, as_of)
            if exposure.state != AT_RISK:
                observation = OBSERVATION_CLOSED
            elif following.received <= as_of:
                observation = OBSERVATION_MATURED
                next_outcome = classify_outcome(following, as_of)

        if exposure_at(data, checkpoint, as_of).state != AT_RISK:
            identification = "not_usable"
            reason = (
                "在评价时点该应约日已不在原协议风险中（暂停、替代覆盖或终止），"
                "不能当成中断结局"
            )
        elif contamination:
            identification = "not_usable"
            reason = contamination
        elif outreach.design != "registered_draw":
            identification = "descriptive_only"
            reason = "非登记分派，没有已知的分配概率，只能描述已联络个案"
        elif not outreach.probabilities or sum(outreach.probabilities) <= 0:
            identification = "descriptive_only"
            reason = "登记分派缺少有效概率"
        elif observation != OBSERVATION_MATURED:
            identification = "not_usable"
            reason = observation
        elif gap_label(next_outcome) not in ("gap", "no_gap"):
            identification = "randomized_itt"
            reason = (
                "登记分派且观察窗已成熟，但下一次结果尚未查齐；"
                "可进入意向性队列，暂不能计入二分效果估计"
            )
        else:
            identification = "randomized_itt"
            reason = "登记分派且观察窗已成熟，可作意向性比较；仍不能证明个案一定避免中断"

        rows.append(
            AssignmentEvidence(
                outreach_id=outreach.id,
                checkpoint_id=checkpoint.id,
                household=data.household_of(checkpoint),
                design=outreach.design,
                arm=outreach.arm,
                target_outcome=target,
                next_checkpoint_id=following.id if following else "",
                next_outcome=next_outcome,
                observation=observation,
                contamination=contamination,
                identification=identification,
                reason=reason,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class StrategyCell:
    design: str
    arm: str
    assignments: int
    eligible: int
    next_known: int
    next_gaps: int
    observed_rate: float | None
    weighted_rate: float | None
    weighted_ci: tuple[float, float] | None


@dataclass(frozen=True)
class StrategyEvaluation:
    as_of: date
    population: int
    cells: tuple[StrategyCell, ...]
    difference: float | None
    difference_ci: tuple[float, float] | None
    notes: tuple[str, ...]
    contaminated: int
    descriptive: int
    target_table: tuple[tuple[str, str, str, int], ...]
    bookkeeping: tuple[tuple[str, str, str, int], ...]


def _arm_probability(outreach: Outreach) -> float | None:
    if not outreach.probabilities:
        return None
    index = {"document": 0, "arrange": 1, "none": 2}[outreach.arm]
    value = outreach.probabilities[index]
    return value if value > 0 else None


def _spell_of(data: PolicyData, checkpoint_id: str, mapping: dict[str, str]) -> str:
    return mapping.get(checkpoint_id, checkpoint_id)


def evaluate_strategy(
    data: PolicyData,
    as_of: date,
    *,
    seed: int = SEED,
    resamples: int = 400,
) -> StrategyEvaluation:
    """评估联络策略。

    主要估计量是“下一次约定服务是否出现已确认缺口”的意向性比较：

    * 只使用登记分派（``registered_draw``）且执行日不早于回单日、观察窗成熟的
      记录；否则联系本身会改变后来查到的结果。
    * 每个分派按其登记概率加权（Horvitz–Thompson），因为分派概率已知；
    * 差异的方向为 ``arrange - none``，区间用按风险经历重抽样的自助法。
    """
    mapping = spell_ids(data, as_of)
    evidence = assignment_evidence(data, as_of)
    by_id = {row.outreach_id: row for row in evidence}

    cells: list[StrategyCell] = []
    target_table: dict[tuple[str, str, str], int] = {}
    usable: dict[str, list[tuple[float, float, str]]] = {"document": [], "arrange": [], "none": []}
    contaminated = 0
    descriptive = 0
    for outreach in data.outreach:
        row = by_id[outreach.id]
        target_table[
            (outreach.arm, row.target_outcome, outreach.result)
        ] = target_table.get((outreach.arm, row.target_outcome, outreach.result), 0) + 1
        if row.contamination:
            contaminated += 1
        if row.identification == "descriptive_only":
            descriptive += 1
        if not row.usable_for_effect:
            continue
        outcome = gap_label(row.next_outcome)
        if outcome not in ("gap", "no_gap"):
            continue
        probability = _arm_probability(outreach)
        if probability is None:
            continue
        usable[outreach.arm].append(
            (
                1.0 if outcome == "gap" else 0.0,
                1.0 / probability,
                _spell_of(data, outreach.checkpoint, mapping),
            )
        )

    population = sum(len(items) for items in usable.values())
    for arm in ("none", "document", "arrange"):
        items = usable[arm]
        if not items:
            cells.append(StrategyCell("registered_draw", arm, 0, 0, 0, 0, None, None, None))
            continue
        gaps = sum(item[0] for item in items)
        weights = sum(item[1] for item in items)
        cells.append(
            StrategyCell(
                design="registered_draw",
                arm=arm,
                assignments=sum(1 for row in evidence if row.arm == arm and row.design == "registered_draw"),
                eligible=len(items),
                next_known=len(items),
                next_gaps=int(gaps),
                observed_rate=gaps / len(items),
                weighted_rate=gaps / weights,
                weighted_ci=wilson_interval(int(gaps), len(items)),
            )
        )

    difference = None
    difference_ci = None
    if usable["arrange"] and usable["none"]:
        arrange_rate = sum(i[0] for i in usable["arrange"]) / sum(i[1] for i in usable["arrange"])
        none_rate = sum(i[0] for i in usable["none"]) / sum(i[1] for i in usable["none"])
        difference = arrange_rate - none_rate
        # 自助法：按风险经历整段重抽样，避免同一经历的前后互相“独立”。
        by_spell: dict[str, list[tuple[str, float, float]]] = {}
        for arm, items in usable.items():
            for value, weight, spell in items:
                by_spell.setdefault(spell, []).append((arm, value, weight))
        spell_keys = sorted(by_spell)
        rng = random.Random(seed)
        replicates: list[float] = []
        for _ in range(resamples):
            sampled = [rng.choice(spell_keys) for _ in spell_keys]
            totals = {arm: [0.0, 0.0] for arm in ("arrange", "none")}
            for spell in sampled:
                for arm, value, weight in by_spell[spell]:
                    if arm in totals:
                        totals[arm][0] += value * weight
                        totals[arm][1] += weight
            if totals["arrange"][1] and totals["none"][1]:
                replicates.append(
                    totals["arrange"][0] / totals["arrange"][1]
                    - totals["none"][0] / totals["none"][1]
                )
        if len(replicates) >= 20:
            replicates.sort()
            difference_ci = (
                replicates[int(0.025 * (len(replicates) - 1))],
                replicates[int(0.975 * (len(replicates) - 1))],
            )

    notes: list[str] = []
    if population:
        notes.append(
            f"意向性比较只纳入 {population} 条登记分派记录；"
            f"另有 {contaminated} 条执行早于回单，结果可能被联系改变，未纳入。"
        )
    else:
        notes.append("没有观察窗成熟且未受联系影响的登记分派记录，无法比较策略效果。")
    if descriptive:
        notes.append(
            f"{descriptive} 条非登记分派只能描述已联络个案，不能当作策略效果证据。"
        )
    if population < 40:
        notes.append("可用记录少于 40 条，区间很宽，结论只能作为方向性参考。")
    if any(len(items) < 10 for items in usable.values()):
        notes.append("至少一个分派组的可用记录少于 10 条，组间差异不稳定。")

    return StrategyEvaluation(
        as_of=as_of,
        population=population,
        cells=tuple(cells),
        difference=difference,
        difference_ci=difference_ci,
        notes=tuple(notes),
        contaminated=contaminated,
        descriptive=descriptive,
        target_table=tuple(
            (arm, outcome, result, count)
            for (arm, outcome, result), count in sorted(target_table.items())
        ),
        bookkeeping=tuple(
            (arm, observation, identification, count)
            for (arm, observation, identification), count in sorted(
                _observation_counts(evidence).items()
            )
        ),
    )


def _observation_counts(
    evidence: tuple[AssignmentEvidence, ...]
) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in evidence:
        key = (row.arm, row.observation, row.identification)
        counts[key] = counts.get(key, 0) + 1
    return counts


def result_semantics(data: PolicyData, as_of: date) -> tuple[dict[str, object], ...]:
    """逐条列出联络结果怎样影响“能否确认结果”与“是否可能避免中断”。

    * ``alt_confirmed``：后来收到另一机构已接手的证明，改变的是结果能否确认；
    * ``booked_new``：联络后新安排了服务，才可能真的避免中断；
    * ``no_change``：没有改变安排或证明。
    """
    rows: list[dict[str, object]] = []
    for row in assignment_evidence(data, as_of):
        outreach = next(
            item for item in data.outreach if item.id == row.outreach_id
        )
        if outreach.result == "alt_confirmed":
            meaning = "改变了结果能否确认：由无服务改记为有替代安排覆盖"
        elif outreach.result == "booked_new":
            meaning = "联络后新安排服务，可能真的避免了中断"
        else:
            meaning = "没有改变安排或证明"
        rows.append(
            {
                "outreach": outreach.id,
                "household": row.household,
                "checkpoint": row.checkpoint_id,
                "result": outreach.result,
                "meaning": meaning,
                "target_outcome": row.target_outcome,
                "next_outcome": row.next_outcome,
                "identification": row.identification,
                "reason": row.reason,
            }
        )
    return tuple(rows)


# --------------------------------------------------------------------------
# 分覆盖与联络策略的回看
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StratumMetrics:
    dimension: str
    value: str
    rows: int
    events: int
    auc: float | None
    brier: float | None
    baseline_brier: float | None


def backtest_strata(
    data: PolicyData,
    result: "Backtest",
    *,
    dimensions: tuple[str, ...] = ("group", "site", "channel", "last_state"),
) -> tuple[StratumMetrics, ...]:
    """把前向回测按覆盖特征与联络策略拆开看。

    维度包括协议的 ``group``/``site``、家庭同意的方式 ``channel``，以及预测时的
    最近一次已确认状态 ``last_state``。分组的样本往往很小，所以每行都同时给出
    事件数和基线 Brier，避免只看 AUC。
    """
    rows: list[StratumMetrics] = []
    prepared: list[dict[str, object]] = []
    for row in result.predictions:
        checkpoint = data.checkpoint(str(row["checkpoint"]))
        episode = data.episode(checkpoint.episode)
        permission = data.permission(episode.household)
        features = features_for(data, checkpoint, date.fromisoformat(str(row["cutoff"])))
        prepared.append(
            {
                "group": episode.group,
                "site": episode.site,
                "channel": permission.channel if permission else "missing",
                "last_state": features.last_state if features else "none",
                "probability": float(row["probability"]),
                "base_rate": float(row["base_rate"]),
                "label": int(row["label"]),
            }
        )
    for dimension in dimensions:
        values = sorted({str(row[dimension]) for row in prepared})
        for value in values:
            subset = [row for row in prepared if str(row[dimension]) == value]
            scores = [float(row["probability"]) for row in subset]
            labels = [int(row["label"]) for row in subset]
            rows.append(
                StratumMetrics(
                    dimension=dimension,
                    value=value,
                    rows=len(subset),
                    events=sum(labels),
                    auc=_auc(scores, labels),
                    brier=_brier(scores, labels),
                    baseline_brier=_brier([float(row["base_rate"]) for row in subset], labels),
                )
            )
    return tuple(rows)
