# Student-only baselines：RuleArena Taxes 与 MTOB

不带 advisor 时，各个 Claude 模型单独作答（"student alone"）在两个测试集上的分数。
测试时间：2026-10-26。模型通过 Vertex AI（`global` 区域）调用。

## 结果

每个测试集跑 5 遍。表中是 5 遍的平均分，括号内是 95% bootstrap 置信区间。

| 模型 | RuleArena Taxes 准确率 (%) | MTOB chrF |
| --- | --- | --- |
| `claude-haiku-4-5` | 66.4 (48.0–84.0) | 40.9 (36.1–46.2) |
| `claude-sonnet-4-5` | 84.0 (68.0–96.0) | 46.2 (40.3–52.6) |
| `claude-sonnet-4-6` | 95.2 (84.0–100.0) | 49.2 (43.2–55.5) |
| `claude-opus-4-6` | 98.4 (92.0–100.0) | 49.6 (43.7–55.7) |

每一遍的分数：

| 模型 | RuleArena（5 遍） | MTOB（5 遍） |
| --- | --- | --- |
| `claude-haiku-4-5` | 76.0 / 76.0 / 52.0 / 68.0 / 60.0 | 41.6 / 41.1 / 40.7 / 41.3 / 40.0 |
| `claude-sonnet-4-5` | 92.0 / 84.0 / 76.0 / 80.0 / 88.0 | 45.0 / 48.2 / 47.1 / 45.8 / 44.9 |
| `claude-sonnet-4-6` | 96.0 / 96.0 / 92.0 / 100.0 / 92.0 | 49.9 / 48.8 / 50.8 / 49.9 / 46.8 |
| `claude-opus-4-6` | 100.0 / 96.0 / 100.0 / 96.0 / 100.0 | 50.9 / 50.2 / 49.6 / 49.9 / 47.4 |

论文（arXiv 2510.02453）中的参考数字：

| | RuleArena Taxes | MTOB |
| --- | --- | --- |
| GPT-4.1 mini 单独作答 | 64.8% | – |
| GPT-5.2 单独作答 | 67.2% | – |
| GPT-4o mini 单独作答 | – | 33.1 |
| Claude Haiku 4.5 单独作答（附录 D） | – | 40.2 |
| 人类专家 | – | 51.6 |

## 设置

所有设置都尽量沿用上游代码。上游的 eval 脚本只评测 "advisor + student" 的组合，
没有单独评测 student 的入口，所以单独作答用的是新增的
[`scripts/eval_student_baseline.py`](../scripts/eval_student_baseline.py)。
它直接调用上游的函数来选题、构造 prompt 和打分。

**RuleArena Taxes**
- 测试集：Level 0 共 100 题，按 `rule_arena/construct_dataset.py` 的方式切分（`random.seed(42)`，75 题训练 / 25 题测试）。
  这样选出的 25 题与上游提交的 `data/rule_arena/validation_gpt-4.1-mini_0.parquet` 完全一致。
- 调用方式：和 `get_initial_response` 一样，只发一条 user 消息（`build_prompt(info_dict)`），没有 system prompt，temperature 用模型默认值（1.0）。
- 打分：`rule_arena.config.compute_score`，从回答中提取 "The total tax owed/overpaid is $X"，与标准答案比较。

**MTOB**
- 测试集：上游提交的 `data/mtob/validation.parquet`，共 50 句 Kalamang→英语。
- Prompt：`BASELINE_PROMPT_TEMPLATE`，即 student prompt 去掉 advisor 建议的部分，参考资料（词典条目和例句）保持不变。
- 调用方式：temperature 0，与 `eval_mtob.py` 对非 GPT-5 学生模型的设置一致。
- 打分：从回答中提取 `Translation:` 之后的内容，计算 chrF。表中分数已 ×100。

**共同设置**
- `max_tokens` 8192。没有开启 extended thinking。
- 每个测试集跑 5 遍，置信区间用 `utils/eval_utils.compute_multi_run_statistics` 计算，与上游 eval 脚本相同。
- 模型名经 `utils/llm.py` 映射为 `vertex_ai/<model>`。

## 数据质量检查

- 1,500 次调用（4 个模型 × (125 + 250)）全部成功，没有 API 报错、限流或截断，`finish_reason` 均为 `stop`。
- RuleArena：每条回答都能提取出答案，0 分都是答案算错，不是提取失败。
- MTOB：只有 Sonnet 4.5 有 1 条（共 250 条）因为写成 `**Translation:**` 加粗格式而没被提取到，按上游规则记 0 分，对平均分的影响约 0.2。

## 注意事项

- **RuleArena 每遍之间波动大**：temperature 是 1.0，而且只有 25 题，置信区间很宽（Haiku 为 48–84%）。拿它和 advisor 做比较时，至少要看 5 遍的平均分。
- **强模型的提升空间很小**：Sonnet 4.6 和 Opus 4.6 在 RuleArena 上已有 95–98%，MTOB 上也接近人类专家的 51.6。要观察 advisor 的作用，Haiku 4.5 是更合适的学生模型。
- **MTOB 要用上游提交的数据**：重新运行 `mtob/construct_dataset.py` 生成的参考资料与上游不同（同样接近的词典条目选法不同，可能与 Python 字符串哈希的随机化有关），用它测出的 Haiku 分数是 42.2，而不是 40.9。
- **不支持的模型**：`claude-opus-4-8` 拒绝任何 `temperature` 参数，现有代码直接用它会在调用失败后静默记 0 分，所以没有纳入本表。

## 复现

先按 [README](../README.md) 配好环境和 Vertex 的环境变量，再解密 MTOB 数据：

```bash
cd advisor_models/mtob/mtob-official && unzip -P kalamang dataset-encrypted-with-password-kalamang.zip && cd -
```

然后对每个模型运行：

```bash
for m in claude-haiku-4-5 claude-sonnet-4-5 claude-sonnet-4-6 claude-opus-4-6; do
  PYTHONPATH=. .venv/bin/python scripts/eval_student_baseline.py rule_arena --student_model $m --num_runs 5
  PYTHONPATH=. .venv/bin/python scripts/eval_student_baseline.py mtob --student_model $m --num_runs 5 \
      --dataset_path data/mtob/validation.parquet
done
```

每条回答和分数会写到 `results/baselines/`。这个目录已加入 `.gitignore`，不提交到仓库，
因为其中包含 MTOB 测试句及其参考翻译，而 MTOB 要求不要把测试数据公开到网上。
