"""定期上门服务的中断风险、联系计划与策略评估。

所有计算只依赖标准库。日期均为北京时间自然日，区间左闭右开。
"""

from .data import PolicyData, load_data

__all__ = ["PolicyData", "load_data"]
