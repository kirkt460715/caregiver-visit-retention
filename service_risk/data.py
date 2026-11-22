"""资料装载、字段校验与只读索引。

所有集合的日期字段在装载时解析为 ``datetime.date``；空串日期表示无期限或未执行，
统一解析为 ``None``，具体含义由使用方按集合说明处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import json

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def parse_date(value: str) -> date | None:
    """把交换文件中的日期转成 ``date``；空串表示没有该日期。"""
    if not value:
        return None
    return date.fromisoformat(value)


def iso(value: date | None) -> str:
    return value.isoformat() if value else ""


@dataclass(frozen=True)
class Episode:
    episode: str
    household: str
    site: str
    group: str
    begins: date
    ends: date


@dataclass(frozen=True)
class Checkpoint:
    id: str
    episode: str
    due: date
    receipt_state: str
    received: date
    original_result: str
    context: str


@dataclass(frozen=True)
class Outreach:
    id: str
    checkpoint: str
    design: str
    probabilities: tuple[float, float, float] | None
    arm: str
    assigned_on: date
    action_day: date | None
    result: str

    @property
    def randomized(self) -> bool:
        return self.design == "registered_draw"


@dataclass(frozen=True)
class ServiceChange:
    id: str
    episode: str
    status: str
    starts: date
    ends: date | None
    recorded_on: date
    new_provider: str

    def covers(self, day: date) -> bool:
        """左闭右开：起日当天生效，止日当天不再生效；无终止日表示持续。"""
        return self.starts <= day and (self.ends is None or day < self.ends)


@dataclass(frozen=True)
class Permission:
    household: str
    channel: str
    starts: date
    status: str
    site: str

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class Callback:
    id: str
    household: str
    site: str
    due: date
    status: str


@dataclass(frozen=True)
class Capacity:
    id: str
    site: str
    week: date
    check_slots: int
    change_slots: int


@dataclass
class PolicyData:
    """一个只读的资料集合，附常用索引。"""

    episodes: tuple[Episode, ...]
    checkpoints: tuple[Checkpoint, ...]
    outreach: tuple[Outreach, ...]
    service_changes: tuple[ServiceChange, ...]
    permissions: tuple[Permission, ...]
    callbacks: tuple[Callback, ...]
    capacity: tuple[Capacity, ...]
    episodes_by_id: dict[str, Episode] = field(init=False, repr=False)
    checkpoints_by_id: dict[str, Checkpoint] = field(init=False, repr=False)
    checkpoints_by_episode: dict[str, tuple[Checkpoint, ...]] = field(init=False, repr=False)
    outreach_by_checkpoint: dict[str, Outreach] = field(init=False, repr=False)
    changes_by_episode: dict[str, tuple[ServiceChange, ...]] = field(init=False, repr=False)
    permission_by_household: dict[str, Permission] = field(init=False, repr=False)
    callbacks_by_household: dict[str, tuple[Callback, ...]] = field(init=False, repr=False)
    capacity_by_site_week: dict[tuple[str, date], Capacity] = field(init=False, repr=False)
    episodes_by_household: dict[str, tuple[Episode, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.episodes_by_id = {row.episode: row for row in self.episodes}
        self.checkpoints_by_id = {row.id: row for row in self.checkpoints}
        self.outreach_by_checkpoint = {row.checkpoint: row for row in self.outreach}
        self.permission_by_household = {row.household: row for row in self.permissions}
        self.capacity_by_site_week = {(row.site, row.week): row for row in self.capacity}

        def group(rows, key):
            grouped: dict[str, list] = {}
            for row in rows:
                grouped.setdefault(getattr(row, key), []).append(row)
            return {name: tuple(items) for name, items in grouped.items()}

        self.checkpoints_by_episode = group(
            sorted(self.checkpoints, key=lambda row: (row.episode, row.due)), "episode"
        )
        self.changes_by_episode = group(
            sorted(self.service_changes, key=lambda row: (row.starts, row.id)), "episode"
        )
        self.callbacks_by_household = group(
            sorted(self.callbacks, key=lambda row: (row.due, row.id)), "household"
        )
        self.episodes_by_household = group(
            sorted(self.episodes, key=lambda row: (row.begins, row.episode)), "household"
        )

    # -- 常用查询 ---------------------------------------------------------
    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        return self.checkpoints_by_id[checkpoint_id]

    def episode(self, episode_id: str) -> Episode:
        return self.episodes_by_id[episode_id]

    def household_of(self, checkpoint: Checkpoint) -> str:
        return self.episodes_by_id[checkpoint.episode].household

    def permission(self, household: str) -> Permission | None:
        return self.permission_by_household.get(household)

    def checkpoints_for_household(self, household: str) -> tuple[Checkpoint, ...]:
        result: list[Checkpoint] = []
        for episode in self.episodes_by_household.get(household, ()):
            result.extend(self.checkpoints_by_episode.get(episode.episode, ()))
        return tuple(sorted(result, key=lambda row: (row.due, row.id)))

    def known_changes(self, episode_id: str, as_of: date) -> tuple[ServiceChange, ...]:
        """截至 ``as_of`` 已登记（recorded_on 不晚于该日）的变更；生效仍按实际期限。"""
        return tuple(
            change
            for change in self.changes_by_episode.get(episode_id, ())
            if change.recorded_on <= as_of
        )

    def validate(self) -> list[str]:
        """返回资料问题列表；空列表表示结构、引用与业务状态一致。"""
        problems: list[str] = []
        seen: set[str] = set()
        for episode in self.episodes:
            if episode.episode in seen:
                problems.append(f"episode 重复：{episode.episode}")
            seen.add(episode.episode)
            if episode.begins >= episode.ends:
                problems.append(f"{episode.episode} 的协议区间不是左闭右开的有效区间")
            checkpoints = self.checkpoints_by_episode.get(episode.episode, ())
            if not checkpoints:
                problems.append(f"{episode.episode} 没有应约记录")
            dues = [row.due for row in checkpoints]
            if len(dues) != len(set(dues)):
                problems.append(f"{episode.episode} 有重复应约日")
            for row in checkpoints:
                if not (episode.begins <= row.due < episode.ends):
                    problems.append(f"{row.id} 的应约日不在协议区间内")
                if row.received < row.due:
                    problems.append(f"{row.id} 的回单早于应约日")
                if row.receipt_state not in {"verified_yes", "verified_no", "unresolved"}:
                    problems.append(f"{row.id} 的回单状态无法识别")
                if row.context not in {"normal", "vacant", "alternate_preexisting"}:
                    problems.append(f"{row.id} 的上下文无法识别")
                expected = {
                    "verified_yes": "fulfilled",
                    "verified_no": "missed",
                    "unresolved": "unknown",
                }[row.receipt_state]
                if row.original_result != expected:
                    problems.append(f"{row.id} 的原服务结果与回单状态不一致")
        for outreach in self.outreach:
            checkpoint = self.checkpoints_by_id.get(outreach.checkpoint)
            if checkpoint is None:
                problems.append(f"{outreach.id} 指向不存在的应约记录")
                continue
            if outreach.design not in {"registered_draw", "discretionary"}:
                problems.append(f"{outreach.id} 的分派方式无法识别")
            if outreach.arm not in {"document", "arrange", "none"}:
                problems.append(f"{outreach.id} 的联络分派无法识别")
            if outreach.assigned_on > checkpoint.due:
                problems.append(f"{outreach.id} 在应约日之后才登记分派")
            if outreach.action_day is not None and outreach.assigned_on > outreach.action_day:
                problems.append(f"{outreach.id} 的执行日早于分派日")
            if outreach.arm == "none" and outreach.action_day is not None:
                problems.append(f"{outreach.id} 未分派联络却有执行日")
            if outreach.arm != "none" and outreach.action_day is None:
                problems.append(f"{outreach.id} 已分派联络却没有执行日")
            if outreach.randomized:
                probabilities = outreach.probabilities
                if probabilities is None or len(probabilities) != 3:
                    problems.append(f"{outreach.id} 缺少登记分派的三个概率")
                elif abs(sum(probabilities) - 1.0) > 1e-6:
                    problems.append(f"{outreach.id} 的登记分派概率之和不为 1")
                elif probabilities[{"document": 0, "arrange": 1, "none": 2}[outreach.arm]] <= 0:
                    problems.append(f"{outreach.id} 的执行分派概率不为正")
            elif outreach.probabilities is not None:
                problems.append(f"{outreach.id} 不是登记分派却登记了概率")
            if outreach.result not in {"booked_new", "alt_confirmed", "no_change"}:
                problems.append(f"{outreach.id} 的联络结果无法识别")
        for change in self.service_changes:
            if change.episode not in self.episodes_by_id:
                problems.append(f"{change.id} 指向不存在的协议区间")
            if change.status not in {
                "temporary_stay",
                "transferred",
                "new_alternate",
                "moved",
            }:
                problems.append(f"{change.id} 的变更状态无法识别")
            if change.ends is not None and change.ends <= change.starts:
                problems.append(f"{change.id} 的变更区间不是左闭右开的有效区间")
            if change.status in {"transferred", "new_alternate"} and not change.new_provider:
                problems.append(f"{change.id} 缺少已经确认的承接机构")
        for permission in self.permissions:
            if permission.status not in {"active", "withdrawn"}:
                problems.append(f"{permission.household} 的联络授权状态无法识别")
            if permission.channel not in {"phone", "text"}:
                problems.append(f"{permission.household} 的联络方式无法识别")
        for callback in self.callbacks:
            if callback.status not in {"confirmed", "withdrawn", "verified"}:
                problems.append(f"{callback.id} 的回拨状态无法识别")
        weeks: set[tuple[str, date]] = set()
        for capacity in self.capacity:
            key = (capacity.site, capacity.week)
            if key in weeks:
                problems.append(f"{capacity.id} 与同站点同周余量重复")
            weeks.add(key)
            if capacity.check_slots < 0 or capacity.change_slots < 0:
                problems.append(f"{capacity.id} 的余量为负数")
        return problems


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        content = json.load(source)
    fields, key, rows = content["fields"], content["key"], content["rows"]
    if key not in fields or len(fields) != len(set(fields)) or not isinstance(rows, list):
        raise ValueError(f"{path.stem} 的字段登记无效")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.keys() != set(fields):
            raise ValueError(f"{path.stem} 有不符合字段登记的记录")
        if not all(isinstance(value, str) for value in row.values()):
            raise ValueError(f"{path.stem} 的字段值必须是字符串")
        if not row[key] or row[key] in seen:
            raise ValueError(f"{path.stem} 的记录键缺失或重复")
        seen.add(row[key])
    return rows


def _probabilities(value: str) -> tuple[float, float, float] | None:
    """登记分派概率按 document、arrange、none 的顺序保存。"""
    if not value:
        return None
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 3:
        raise ValueError("登记分派概率必须是三项")
    return parts  # type: ignore[return-value]


def load_data(directory: Path | str = DEFAULT_DATA, strict: bool = True) -> PolicyData:
    """装载全部集合；``strict`` 为真时遇到资料问题即报错。"""
    directory = Path(directory)
    rows = {path.stem: _read_rows(path) for path in sorted(directory.glob("*.json"))}
    required = {
        "episodes",
        "checkpoints",
        "outreach",
        "service_changes",
        "permissions",
        "callback",
        "capacity",
    }
    missing = required - rows.keys()
    if missing:
        raise ValueError(f"缺少集合：{', '.join(sorted(missing))}")

    data = PolicyData(
        episodes=tuple(
            Episode(
                episode=row["episode"],
                household=row["household"],
                site=row["site"],
                group=row["group"],
                begins=date.fromisoformat(row["begins"]),
                ends=date.fromisoformat(row["ends"]),
            )
            for row in rows["episodes"]
        ),
        checkpoints=tuple(
            Checkpoint(
                id=row["id"],
                episode=row["episode"],
                due=date.fromisoformat(row["due"]),
                receipt_state=row["receipt_state"],
                received=date.fromisoformat(row["received"]),
                original_result=row["original_result"],
                context=row["context"],
            )
            for row in rows["checkpoints"]
        ),
        outreach=tuple(
            Outreach(
                id=row["id"],
                checkpoint=row["checkpoint"],
                design=row["design"],
                probabilities=_probabilities(row["probabilities"]),
                arm=row["arm"],
                assigned_on=date.fromisoformat(row["assigned_on"]),
                action_day=parse_date(row["action_day"]),
                result=row["result"],
            )
            for row in rows["outreach"]
        ),
        service_changes=tuple(
            ServiceChange(
                id=row["id"],
                episode=row["episode"],
                status=row["status"],
                starts=date.fromisoformat(row["starts"]),
                ends=parse_date(row["ends"]),
                recorded_on=date.fromisoformat(row["recorded_on"]),
                new_provider=row["new_provider"],
            )
            for row in rows["service_changes"]
        ),
        permissions=tuple(
            Permission(
                household=row["household"],
                channel=row["channel"],
                starts=date.fromisoformat(row["starts"]),
                status=row["status"],
                site=row["site"],
            )
            for row in rows["permissions"]
        ),
        callbacks=tuple(
            Callback(
                id=row["id"],
                household=row["household"],
                site=row["site"],
                due=date.fromisoformat(row["due"]),
                status=row["status"],
            )
            for row in rows["callback"]
        ),
        capacity=tuple(
            Capacity(
                id=row["id"],
                site=row["site"],
                week=date.fromisoformat(row["week"]),
                check_slots=int(row["check_slots"]),
                change_slots=int(row["change_slots"]),
            )
            for row in rows["capacity"]
        ),
    )
    problems = data.validate()
    if strict and problems:
        raise ValueError("资料校验失败：\n" + "\n".join(problems))
    return data
