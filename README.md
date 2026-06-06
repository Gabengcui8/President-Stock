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

# 拓宽文本源：抓公开新闻，并尽量抓取整篇正文；默认覆盖 Trump、市场、关税、中国、通胀、能源等主题
# 默认抓最近约 90 天；可以用 --start-date/--end-date 指定日期范围，实际覆盖以 GDELT 返回为准
python -m spy_trump_model fetch-gdelt-news --out data/raw/news.csv --max-records 100 --fetch-article-text --max-article-fetches 50 --sleep-seconds 4 --article-sleep-seconds 2

# 可选：自定义新闻查询；label=query 可以重复传
python -m spy_trump_model fetch-gdelt-news --out data/raw/news.csv --fetch-article-text --max-article-fetches 30 --sleep-seconds 4 --query 'tariffs="Donald Trump" tariffs' --query 'oil="Donald Trump" oil energy'

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

# 拓宽信号来源：合并多个文本源，生成整篇文本向量、情绪、解释性主题标签和市场状态
python -m spy_trump_model signal-expansion --texts data/raw/trump_speeches.csv data/raw/news.csv --out data/processed/expanded_signals.csv --start 2021-01-01

# 纸面观察账本：只记录文本观察日后 1/3/5 个交易日各资产实际表现，不自动下单
python -m spy_trump_model paper-ledger --signals data/processed/expanded_signals.csv --tickers SPY QQQ GLD TLT USO XLE XLI XLF SMH FXI --horizons 1 3 5
```

`keyword-impact` 默认只分析 `2021-01-01` 之后的数据；可以用 `--analysis-start YYYY-MM-DD` 覆盖。默认策略候选还要求训练期 `abs(t-stat) >= 1.5`、事件前后两半方向一致，并且满足 `--min-keyword-days` 和 `--min-independent-events`。独立事件会按 horizon 去掉重叠窗口，例如 3 日 horizon 下连续 3 个关键词日只算 1 个独立事件。默认成本是 5 bps，按一次进出场收取两边成本。

`keyword-impact` 默认排除没有 `datetime` 的未知时间戳发言。盘前/盘中发言假设最早在信号日收盘成交；盘后/周末发言会先对齐到下一个交易日，再按该日收盘成交。只有做数据审计时才建议加 `--include-unknown-time`。

`summary.csv` 是完整研究表，`selected.csv` 是通过训练期过滤的候选表，`robust_selected.csv` 更严格：默认只看 `vol_regime=all`，以 3 日 horizon 为主，并要求 1 日和 5 日同方向，而且 1/3/5 三个 horizon 的训练期和测试期独立事件数都分别达到 `--min-robust-train-independent-events` 和 `--min-robust-test-independent-events`。

如果有信号进入 `robust_selected.csv`，程序还会自动做留一事件法诊断。`robust_event_returns.csv` 列出每个测试期独立事件的净收益和累计净收益；`robust_jackknife.csv` 逐个删除事件后重算净收益、event Sharpe 和最大回撤；`robust_jackknife_summary.csv` 汇总最差留一结果。`jackknife_fragile=True` 表示删掉某个事件后，总收益或 event Sharpe 会从正数塌到非正数，或者最大单事件贡献/收益缩水过高，或者最差留一 Sharpe 低于诊断地板。它是风险提示，不是新的筛选阈值，具体触发原因看 `jackknife_fragility_reasons`。

`signal-expansion` 用来解决离散关键词事件太少的问题。它接收一个或多个 CSV 文本源，格式同样建议使用 `date,datetime,title,source,source_type,text`。核心输出是 `whole_text_vec_*`：每天全部文章/帖子正文拼接后的整篇文本哈希向量，模型可以从完整文本的词组结构里学习，而不是只数关键词。`theme_*` 仍会保留，但只应作为解释和审计标签；同时还会输出情绪、来源计数、60 日 surprise，以及 SPY/QQQ/GLD/TLT/USO/UUP/VIX/TNX 等市场状态。这个表适合进入 paper trading 和后续模型，不应直接当实盘信号。

`fetch-gdelt-news` 用来把 Trump 自己发言之外的公开新闻纳入文本源，默认查询 Trump+市场、Trump+贸易/中国、Trump+通胀/利率、Trump+能源、Trump+美元、Trump+地缘政治、Trump+税收监管、Trump+边境移民等组合。加 `--fetch-article-text` 后会访问新闻 URL 并尽量抽取正文，失败时退回标题，不会让整批任务中断。为了避免访问过多 URL 被限流，默认最多抓 50 篇正文，可以用 `--max-article-fetches` 调整，`-1` 表示不限制。GDELT API 如果返回 429，程序会按 `Retry-After` 或指数退避自动等待重试；长区间建议加 `--sleep-seconds 4`，更严重时用 `--gdelt-retry-backoff-seconds 60 --gdelt-retry-attempts 6`。它写入 `data/raw/news.csv`，之后用 `signal-expansion --texts data/raw/trump_speeches.csv data/raw/news.csv` 合并。注意，新闻正文是“市场如何报道/解读 Trump”的代理变量，不等于 Trump 本人发言，所以后面必须继续用 paper ledger 和样本外检验分开看。

`paper-ledger` 会从 `expanded_signals.csv` 生成前向观察账本，默认只保留 `text_item_count > 0` 的日期，并默认从下一交易日收盘开始计算 1/3/5 日 forward return，避免日级文本把盘后信息混进当天收盘。它的作用是记录和复盘，不是自动交易器；最新几天因为未来价格还没出现，forward return 会是空值。

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
- `data/raw/news.csv`：GDELT 等公开新闻标题/消息源数据
- `data/processed/model_dataset.csv`：建模用数据
- `outputs/signals.csv`：每日预测概率和信号
- `outputs/metrics.json`：回测指标
- `outputs/assets/summary.csv`：多资产比较结果
- `outputs/keyword_impact/summary.csv`：关键词影响的样本外验证结果
- `outputs/keyword_impact/selected.csv`：只包含训练期通过过滤的候选信号
- `outputs/keyword_impact/robust_selected.csv`：独立事件数、3 日主窗口和 1/5 日一致性都通过的更严格候选
- `outputs/keyword_impact/robust_event_returns.csv`：稳健候选的每个测试期独立事件收益
- `outputs/keyword_impact/robust_jackknife.csv`：逐个删除独立事件后的留一诊断
- `outputs/keyword_impact/robust_jackknife_summary.csv`：留一诊断的摘要，检查信号是否被少数事件撑住
- `outputs/keyword_impact/splits.csv`：每个资产的训练/测试日期切分，确认没有重合
- `data/processed/expanded_signals.csv`：连续主题强度、新闻/发言来源、情绪和市场状态信号
- `outputs/paper_trading/ledger.csv`：纸面观察账本，记录每个文本观察日和未来 1/3/5 日资产表现
- `outputs/paper_trading/summary.csv`：纸面观察账本的资产/窗口汇总

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
