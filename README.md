# Trump Speech SPY Model

一个用于研究“特朗普公开发言”和 SPY 后续收益关系的最小可运行项目。

> 说明：这不是投资建议。模型默认只做研究和回测，实盘前需要加入交易成本、滑点、仓位限制、风控和更严格的样本外验证。

## 思路

模型把每日特朗普发言聚合成事件特征，再和 SPY 日线数据对齐：

- SPY 数据：用 `yfinance` 下载一次到 `data/raw/SPY.csv`，之后本地存在就直接读取。
- 发言/帖子数据：优先读取 `data/raw/trump_speeches.csv`。你可以手动维护，也可以抓取 Trump's Truth 归档、Truth Social 公开帖子和白宫公开视频/remarks 条目。
- 特征：每日发言数量、文本长度、情绪分数、重点词频，以及文本哈希特征。文本哈希会让模型从词和短语里学习，不只依赖手写关键词。
- 时间对齐：带时间戳的帖子按美东时间拆成盘前、盘中、盘后、周末；盘后和非交易日帖子会对齐到下一个 SPY 交易日。
- 标签：当天发言预测 SPY 下一交易日涨跌，避免把未来价格泄漏进训练。
- 模型：逻辑回归基础版，时间序列 walk-forward 回测，输出信号和指标。

## 云服务器部署指令

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

mkdir -p ~/trump-spy-model
cd ~/trump-spy-model

# 把本项目文件上传到这个目录后执行：
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 第一次下载 SPY 到本地；以后存在 data/raw/SPY.csv 会直接复用
python -m spy_trump_model download-spy --ticker SPY --start 2015-01-01

# 推荐：抓取 Trump's Truth 归档 RSS，云服务器更稳定
python -m spy_trump_model fetch-trumpstruth --start-date 2022-02-01

# 可选：直接抓 @realDonaldTrump 的 Truth Social 公开接口
# 如果云服务器 IP 被 Truth Social 拒绝，可能会返回 403
python -m spy_trump_model fetch-truthsocial --handle realDonaldTrump --max-pages 10

# 可选：补充白宫公开视频/remarks 页面条目
python -m spy_trump_model fetch-whitehouse --pages 5

# 如果你自己有发言 CSV，放到 data/raw/trump_speeches.csv
# 必需列：date,text；推荐列：title,source

# 训练并回测
python -m spy_trump_model train --min-train-days 252

# 让模型自己比较多个资产，而不是手写“关键词 -> 行业”
python -m spy_trump_model compare-assets --tickers SPY QQQ XLE XLI XLF SMH FXI TLT USO GLD --cost-bps 1

# 用训练期真实收益学习“关键词/主题 -> 资产”的方向和强度，再只在测试期验证
python -m spy_trump_model keyword-impact --tickers SPY QQQ XLE XLI XLF SMH FXI TLT USO GLD --train-fraction 0.7 --min-keyword-days 20

# 更严格：训练期独立事件样本前后两半方向必须一致，只保留做多方向、t-stat 至少 1.5
python -m spy_trump_model keyword-impact --tickers SPY QQQ SMH XLI --train-fraction 0.7 --min-keyword-days 5 --min-independent-events 5 --min-abs-t-stat 1.5 --allowed-direction long

# 信号有效期比较：同时验证 1/3/5 个交易日，并按 SPY 20 日波动率分层
python -m spy_trump_model keyword-impact --tickers SPY QQQ XLE XLI XLF SMH FXI TLT USO GLD --horizons 1 3 5 --train-fraction 0.7 --min-keyword-days 5

# 只做探索、想看所有弱信号时才把 t-stat 门槛降到 0；不要把这个输出直接当交易候选
python -m spy_trump_model keyword-impact --tickers SPY QQQ XLE XLI XLF SMH FXI TLT USO GLD --horizons 1 3 5 --train-fraction 0.7 --min-keyword-days 5 --min-abs-t-stat 0
```

`keyword-impact` 默认只分析 `2021-01-01` 之后的数据；可以用 `--analysis-start YYYY-MM-DD` 覆盖。默认策略候选还要求训练期 `abs(t-stat) >= 1.5`、事件前后两半方向一致，并且满足 `--min-keyword-days` 和 `--min-independent-events`。独立事件会按 horizon 去掉重叠窗口，例如 3 日 horizon 下连续 3 个关键词日只算 1 个独立事件。默认成本是 5 bps，按一次进出场收取两边成本。

`keyword-impact` 默认排除没有 `datetime` 的未知时间戳发言。盘前/盘中发言假设最早在信号日收盘成交；盘后/周末发言会先对齐到下一个交易日，再按该日收盘成交。只有做数据审计时才建议加 `--include-unknown-time`。

`summary.csv` 是完整研究表，`selected.csv` 是通过训练期过滤的候选表，`robust_selected.csv` 更严格：默认只看 `vol_regime=all`，以 3 日 horizon 为主，并要求 1 日和 5 日同方向，测试期独立事件数达到 `--min-test-independent-events`。

定时每天美股收盘后运行，例如服务器时区为 UTC，约等于美东 18:30：

```bash
crontab -e
```

加入：

```cron
30 23 * * 1-5 cd ~/trump-spy-model && . .venv/bin/activate && mkdir -p logs && python -m spy_trump_model download-spy --ticker SPY --start 2015-01-01 --update && python -m spy_trump_model fetch-trumpstruth --start-date 2022-02-01 && python -m spy_trump_model fetch-whitehouse --pages 3 && python -m spy_trump_model train >> logs/run.log 2>&1
```

## 文件输出

- `data/raw/SPY.csv`：SPY 本地缓存
- `data/raw/trump_speeches.csv`：特朗普发言数据
- `data/processed/model_dataset.csv`：建模用数据
- `outputs/signals.csv`：每日预测概率和信号
- `outputs/metrics.json`：回测指标
- `outputs/assets/summary.csv`：多资产比较结果
- `outputs/keyword_impact/summary.csv`：关键词影响的样本外验证结果
- `outputs/keyword_impact/selected.csv`：只包含训练期通过过滤的候选信号
- `outputs/keyword_impact/robust_selected.csv`：独立事件数、3 日主窗口和 1/5 日一致性都通过的更严格候选
- `outputs/keyword_impact/splits.csv`：每个资产的训练/测试日期切分，确认没有重合

## 如果输出里 speeches 是 0 行

先单独运行：

```bash
python -m spy_trump_model fetch-trumpstruth --start-date 2022-02-01
python -m spy_trump_model fetch-truthsocial --handle realDonaldTrump --max-pages 10
python -m spy_trump_model fetch-whitehouse --pages 10
```

确认 `data/raw/trump_speeches.csv` 里有数据后再训练。新版代码会在发言文件没有有效行时直接停止训练，避免只用 SPY 滞后收益跑出误导性的结果。

## 发言 CSV 格式

```csv
date,datetime,title,source,source_type,text
2026-05-08,2026-05-08T14:30:00Z,Truth Social @realDonaldTrump,https://truthsocial.com/@realDonaldTrump/posts/...,truthsocial,"post text here"
```

`date` 可以是 `YYYY-MM-DD`，也可以是更完整的时间戳；模型会按日期聚合。
