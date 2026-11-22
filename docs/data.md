# 资料说明

各集合保存在 `data/`，JSON 对象的 `fields` 为字段列表，`key` 为集合主键，`rows` 为记录数组。各记录值均为字符串，空串含义以各集合的说明为准。

## episodes

与家庭约定的服务期间、负责站点及事先可用的分组。

主键为 `episode`，字段为 `episode`、`household`、`site`、`group`、`begins`、`ends`。

## checkpoints

历史应约与查证状态。证据、原服务结果和别处已提供的覆盖不等同，后续查证可以确认此前安排。

主键为 `id`，字段为 `id`、`episode`、`due`、`receipt_state`、`received`、`original_result`、`context`。

## outreach

每个被考虑联系的个案对应一种分派。document 是只查证，arrange 是可主动改约。action_day 为空表示未执行。

主键为 `id`，字段为 `id`、`checkpoint`、`design`、`probabilities`、`arm`、`assigned_on`、`action_day`、`result`。

## service_changes

已有替代、暂停、转住和终止日期。new_provider 指已经确认的独立服务安排。

主键为 `id`，字段为 `id`、`episode`、`status`、`starts`、`ends`、`recorded_on`、`new_provider`。

## permissions

当前允许联络的渠道与最先有权处理的站点；withdrawn 不可联系。

主键为 `household`，字段为 `household`、`channel`、`starts`、`status`、`site`。

## callback

已有联系计划、尽量避免重复的改约范围。

主键为 `id`，字段为 `id`、`household`、`site`、`due`、`status`。

## capacity

先查证和实际改约所需的站点余量。字段是次数，已承诺回拨另占一个同类名额。

主键为 `id`，字段为 `id`、`site`、`week`、`check_slots`、`change_slots`。
