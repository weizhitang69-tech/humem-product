# Structured Memory Fidelity Evaluation

- Samples: 2000
- Seed: 20260511

## Accuracy By Simulation Mode

| Mode | Accuracy | Correct / Total | Main Error Counts |
| --- | ---: | ---: | --- |
| `full_role_position` | 100.00% | 2000 / 2000 | - |
| `no_position` | 91.80% | 1836 / 2000 | missing_position=164 |
| `value_bag` | 63.95% | 1279 / 2000 | missing_position=164, wrong_role=557 |
| `noisy_full_role_position` | 85.45% | 1709 / 2000 | missing_target=137, wrong_position=24, wrong_value=130 |

## What This Means

- `full_role_position` estimates whether the structured subtags themselves preserve the facts needed to answer.
- `no_position` shows the loss from removing event order/stage when the same role appears more than once.
- `value_bag` approximates a weak prompt where the model treats subtags as loose text and may confuse roles.
- `noisy_full_role_position` estimates the combined risk of structured recall plus extraction errors.

## Error Examples

### value_bag

- `structured-fidelity-0001` 现在想做什么？ expected `想先比较两家银行`, predicted `想申请分期额度买搬家押金` (wrong_role).
- `structured-fidelity-0002` 从哪里出发？ expected `上海`, predicted `星河青旅` (wrong_role).
- `structured-fidelity-0003` 下一步要做什么？ expected `明天补一组回归测试`, predicted `把召回阈值调高` (wrong_role).
- `structured-fidelity-0008` 和谁一起？ expected `陈晨`, predicted `李然` (wrong_role).
- `structured-fidelity-0014` 到哪里去？ expected `南京`, predicted `北站公寓` (wrong_role).

### noisy_full_role_position

- `structured-fidelity-0012` 讨论什么主题？ expected `模型成本`, predicted `` (missing_target).
- `structured-fidelity-0018` 会上做了什么？ expected `要求先做小流量验证`, predicted `暂时没申请` (wrong_value).
- `structured-fidelity-0020` 从哪里出发？ expected `深圳`, predicted `北京` (wrong_value).
- `structured-fidelity-0024` 讨论什么主题？ expected `续费风险`, predicted `降噪耳机` (wrong_value).
- `structured-fidelity-0026` 晚上住在哪里？ expected `星河青旅`, predicted `` (missing_target).

### no_position

- `structured-fidelity-0060` 负责人是谁？ expected `陈晨`, predicted `阿哲` (missing_position).
- `structured-fidelity-0064` 药放在哪里？ expected `床头柜第二层`, predicted `社区诊所` (missing_position).
- `structured-fidelity-0070` 药放在哪里？ expected `厨房白色药盒`, predicted `南山门诊` (missing_position).
- `structured-fidelity-0076` 药放在哪里？ expected `书桌左边抽屉`, predicted `社区诊所` (missing_position).
- `structured-fidelity-0078` 负责人是谁？ expected `阿哲`, predicted `Maya` (missing_position).
