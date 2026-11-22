# 定期上门服务的中断风险与联系策略验证

服务协议、查证与外访资料。应约日按北京时间。每个协议区间都左闭右开，可以在结束后开新协议。出联络名单不是服务承诺。registered_draw 行的三个概率按 document、arrange、none 的顺序保存，discretionary 没有登记分派概率。联系结果发生在之后收到回单时，只允许用于当时已有的证据。

资料均为合成记录，采用 UTF-8 编码。交换字段、标识和状态见[资料说明](docs/data.md)。

项目使用 Python 3.11.2，资料查看与检测只依赖标准库。在项目目录运行以下命令，可查看各集合的记录数，并检查字段、重复键与跨表引用。

```sh
python3 inspect_records.py
python3 -m unittest discover -s tests -v
```
