# Trump Speech SPY Model

一个用于研究“特朗普公开发言”和 SPY 后续收益关系的最小可运行项目。

> 说明：这不是投资建议。模型默认只做研究和回测，实盘前需要加入交易成本、滑点、仓位限制、风控和更严格的样本外验证。

## 思路

模型把每日特朗普发言聚合成事件特征，再和 SPY 日线数据对齐：

- SPY 数据：用 `yfinance` 下载一次到 `data/raw/SPY.csv`，之后本地存在就直接读取。
- 发言/帖子数据：优先读取 `data/raw/trump_speeches.csv`。你可以手动维护，也可以抓取 Trump's Truth 归档、Truth Social 公开帖子和白宫公开视频/remarks 条目。
- 特征：每日发言数量、文本长度、情绪分数、重点词频，例如 tariff、china、fed、rate、oil、war、tax、regulation。
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
```

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
