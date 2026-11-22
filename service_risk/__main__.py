"""命令行入口：预测、周计划、督导简报与策略评估。

用法示例（日期均为北京时间，周计划按周一起算）：

    python3 -m service_risk predict 2026-07-01
    python3 -m service_risk plan 2026-07-06
    python3 -m service_risk briefing 2026-07-06
    python3 -m service_risk case H064 2026-07-06
    python3 -m service_risk backtest 2026-02-23 2026-09-01
    python3 -m service_risk strategy 2026-09-10

输出为 UTF-8 JSON；加 ``--table`` 改为便于终端阅读的纯文本。
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import sys

from .data import load_data
from .evaluation import backtest, backtest_strata, exposure_report
from .planning import build_weekly_plan
from .risk import fit_logistic, predict_week, training_rows
from .supervisor import case_review, strategy_review, weekly_briefing


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "（无）"
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    lines = ["  ".join(column.ljust(widths[column]) for column in columns)]
    lines.append("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        lines.append("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"日期应为 YYYY-MM-DD：{value}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="定期上门服务的中断风险与联系策略")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("predict", help="某个工作日的应约中断预测")
    command.add_argument("as_of", type=_date)
    command.add_argument("--horizon", type=int, default=7)
    command.add_argument("--table", action="store_true")

    command = sub.add_parser("plan", help="一周联系计划")
    command.add_argument("week", type=_date)
    command.add_argument("--as-of", type=_date, default=None)
    command.add_argument("--table", action="store_true")

    command = sub.add_parser("briefing", help="督导简报（三个名单与证据强度）")
    command.add_argument("week", type=_date)
    command.add_argument("--as-of", type=_date, default=None)
    command.add_argument("--table", action="store_true")

    command = sub.add_parser("case", help="按家庭查看个案")
    command.add_argument("household")
    command.add_argument("as_of", type=_date)
    command.add_argument("--table", action="store_true")

    command = sub.add_parser("backtest", help="预警前向回测")
    command.add_argument("start", type=_date)
    command.add_argument("end", type=_date)

    command = sub.add_parser("strategy", help="联络策略评估")
    command.add_argument("as_of", type=_date)

    command = sub.add_parser("exposure", help="某窗口的暴露与结果分布")
    command.add_argument("as_of", type=_date)
    command.add_argument("--horizon", type=int, default=7)

    args = parser.parse_args(argv)
    data = load_data()

    if args.command == "predict":
        model = fit_logistic(training_rows(data, args.as_of))
        rows = [prediction.as_dict() for prediction in predict_week(
            data, args.as_of, horizon_days=args.horizon, model=model
        )]
        if args.table:
            print(_table(rows, ["checkpoint", "household", "due", "probability", "band", "state", "limitation"]))
        else:
            print(_json({"as_of": args.as_of.isoformat(), "rows": rows}))
        return 0

    if args.command == "plan":
        plan = build_weekly_plan(data, args.week, as_of=args.as_of)
        rows = [item.as_dict() for item in plan.items]
        if args.table:
            print(_table(rows, ["kind", "household", "checkpoint", "due", "site", "channel", "probability", "reason"]))
            print()
            print(_table([slot.__dict__ for slot in plan.slots], ["site", "check_total", "check_used", "change_total", "change_used"]))
            for note in plan.notes:
                print("注：" + note)
        else:
            print(_json({"week": plan.week.isoformat(), "as_of": plan.as_of.isoformat(), "items": rows,
                         "slots": [slot.__dict__ for slot in plan.slots], "notes": list(plan.notes)}))
        return 0

    if args.command == "briefing":
        briefing = weekly_briefing(data, args.week, as_of=args.as_of)
        if args.table:
            for section in briefing.sections:
                print(f"== {section.title}")
                print(section.note)
                print(_table([{key: value for key, value in row.items()} for row in section.rows],
                             ["household", "checkpoint", "due", "probability", "band", "planned", "identification", "reason"]))
                print()
            for note in briefing.support:
                print("注：" + note)
        else:
            print(_json({
                "week": briefing.week.isoformat(),
                "as_of": briefing.as_of.isoformat(),
                "sections": [
                    {"key": section.key, "title": section.title,
                     "note": section.note, "rows": list(section.rows)}
                    for section in briefing.sections
                ],
                "strategy": strategy_review(data, briefing.as_of),
                "support": list(briefing.support),
            }))
        return 0

    if args.command == "case":
        review = case_review(data, args.household, args.as_of)
        payload = {
            "household": review.household,
            "as_of": review.as_of.isoformat(),
            "channel": review.channel,
            "permission": review.permission,
            "site": review.site,
            "episode_timeline": list(review.episode_timeline),
            "predictions": [row.as_dict() for row in review.predictions],
            "evidence": list(review.evidence),
            "spells": list(review.spell_ids),
            "limitations": list(review.limitations),
        }
        if args.table:
            print(_table([row.as_dict() for row in review.predictions],
                         ["checkpoint", "due", "probability", "band", "state", "planned",
                          "outcome_now", "outcome_final"]))
            for row in review.predictions:
                print(f"  {row.checkpoint_id}：{row.planned_reason}")
            if review.evidence:
                print()
                print(_table(list(review.evidence),
                             ["outreach", "arm", "result", "identification", "observation"]))
            for text in review.limitations:
                print("限制：" + text)
        else:
            print(_json(payload))
        return 0

    if args.command == "backtest":
        result = backtest(data, args.start, args.end)
        print(_json({
            "summary": result.summary(),
            "calibration": [bin.__dict__ for bin in result.calibration],
            "strata": [
                {
                    "dimension": row.dimension,
                    "value": row.value,
                    "rows": row.rows,
                    "events": row.events,
                    "auc": row.auc,
                    "brier": row.brier,
                    "baseline_brier": row.baseline_brier,
                }
                for row in backtest_strata(data, result)
            ],
        }))
        return 0

    if args.command == "strategy":
        print(_json(strategy_review(data, args.as_of)))
        return 0

    if args.command == "exposure":
        rows = [
            {
                "checkpoint": row.checkpoint_id,
                "household": row.household,
                "due": row.due.isoformat(),
                "exposure": row.exposure,
                "reason": row.exposure_reason,
                "outcome_now": row.outcome_now,
                "outcome_final": row.outcome_final,
                "note": row.note,
            }
            for row in exposure_report(data, args.as_of, horizon_days=args.horizon)
        ]
        print(_json(rows))
        return 0

    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    sys.exit(main())
