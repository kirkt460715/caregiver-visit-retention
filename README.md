# 定期上门服务的中断风险与联系策略验证

服务协议、查证与外访资料。应约日按北京时间。每个协议区间都左闭右开，可以在结束后开新协议。出联络名单不是服务承诺。registered_draw 行的三个概率按 document、arrange、none 的顺序保存，discretionary 没有登记分派概率。联系结果发生在之后收到回单时，只允许用于当时已有的证据。

资料均为合成记录，采用 UTF-8 编码。交换字段、标识和状态见[资料说明](docs/data.md)，时点口径、风险经历、评估与计划方法见[方法说明](docs/method.md)。

## 能力

`service_risk` 只用标准库，提供：

* 某个工作日的应约中断预测（只用当时能得到的记录）；
* 已确认转住、暂停、机构替代按实际期限生效，迁走/转出后不再计算原协议中断风险；
* 把“各渠道证实无服务”“各渠道尚未查齐”“未来还没到约定日期”分开；
* 按风险经历划分同一家庭的反复开通、暂停与重新联系；
* 前向回测（同一经历整段留出）与按登记分派的意向性策略评估；
* 遵守授权、承接机构优先权、班期额度与约定回拨时间的每周联系计划；
* 督导个案视图、三个不同的名单与低支持限制说明。

## 常用命令

```sh
python3 inspect_records.py
python3 -m unittest discover -s tests -v

python3 -m service_risk predict 2026-07-01 --table
python3 -m service_risk plan 2026-07-06 --table
python3 -m service_risk briefing 2026-07-06 --table
python3 -m service_risk case H064 2026-07-06
python3 -m service_risk backtest 2026-02-23 2026-09-01
python3 -m service_risk strategy 2026-09-10
```

项目使用 Python 3.11.2，资料查看、检测与分析只依赖标准库。
