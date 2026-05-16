# SCUTQUANT Terminal（Worldquant Style backtest web terminal）
## 总纲
目前的需求是实现一个和Worldquant风格的回测网页终端，用户可以在网页上输入opertaors.py中的算子组合成因子表达式进行回测，查看回测结果。
## 数据&账户管理层
本地原始数据使用dataset/all_historical_daily_data.csv,包含20100101 -- 20251101时间段内全A股市场股票的日频数据。
数据格式如下所示
ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
000001.SZ,20100104,24.52,24.58,23.68,23.71,24.37,-0.66,-2.71,241922.76,580249.4715
000002.SZ,20100104,10.85,10.87,10.6,10.6,10.81,-0.21,-1.94,969832.53,1034344.7604
000005.SZ,20100104,6.01,6.05,5.91,5.99,6.02,-0.03,-0.5,223582.22,133478.4207
000006.SZ,20100104,11.33,11.35,11.11,11.12,11.33,-0.21,-1.85,62998.05,70548.5572
000007.SZ,20100104,7.09,7.09,6.9,6.93,7.03,-0.1,-1.42,25901.79,18101.4161
使用sqlite存储原始数据，数据库文件为dataset/all_historical_daily_data.db
采用fastapi框架搭建后端服务，提供因子表达式的回测接口。要求有用户登录注册功能。
在前端上，选择数据而不进行表达式输入阶段，用户可以在下拉框中选择需要回测的股票代码、时间窗口、并设置初始资金、单支股票占比上限、t+0/t+1限制选项，只能开多/只能开空/开多空均可选项。
用户登录注册功能可以参考fastapi的官方文档。
## 前端/UI总纲
采用react框架搭建前端界面，包含登录注册页面、账户信息界面、回测页面，回测界面包含表达式输入终端框、下拉框、回测结果展示框，其中表达式输入终端框类似Worldquant的因子表达式输入框（命令行样式），下拉框包含需要回测的股票代码、时间窗口、初始资金、单支股票占比上限。
回测结果展示框展示回测结果，包含因子表达式、股票代码、时间窗口、初始资金、单支股票占比上限、回测结果（包括年化收益率、最大回撤、夏普比率、年化波动率）。回测结果展示框在回测完成后自动弹出，也可以提前勾选按钮展示覆盖在表达式输入终端框右半部分。
## 因子表达式层
用户可以在前端上输入opertaors.py中的算子组合成因子表达式，operators.py中的算子较为完备，支持dataframe/dataframegroupby格式输入，输出统一为dataframe，也通过装饰器实现了输入输出端自动化inf to nan处理以及时间窗口列表参数传入。需要说明的是只在ipynb中用operators.py中的算子进行回测时，如果只对dataframe中某一列进行回测，传入数据参数要使用类似df["close"]的格，在终端窗口中需要实现为可以直接使用open,high,low,close,pre_close,change,pct_chg,vol,amount等字段作为输入参数。
输入表达式并点击回测按钮，后端服务会根据用户输入的因子表达式进行回测，返回回测结果。
回测结果展示框（简化为“图表 + 指标表”）：
1) 基本信息：因子表达式、股票池/代码范围、时间窗口、初始资金。
2) 图表选项（7–8种，可切换）：累计净值（PnL）、水下回撤（Drawdown）、月度收益热力图、收益分布（Histogram/KDE）、滚动Sharpe、滚动波动率、IC与累计IC、分位收益（Deciles多空spread）。
3) 指标表（约10–15项）：年化收益、年化波动率、Sharpe、Sortino、最大回撤、Calmar、胜率、盈亏比、平均日收益、收益偏度/峰度、VaR(95%/99%)与ES、信息比（IR）、日均换手、平均持有期、成本净效应（按默认成本假设）。
说明：图表用于直观趋势与稳定性评估；指标表用于全面量化策略质量与风险。
图不要用matplotlib那种样式，用常见股票交易软件的那些可以选择时间窗口滑动细看的样式。

## 技术栈与架构选型（精简版）
前端：
- React + TypeScript + Vite + React Router（轻量快速）
- 数据请求：axios；缓存可用 React Query（可选，先不强依赖）
- UI组件：Ant Design；表格先用 AntD Table（不引入 AG Grid）
- 图表：默认 Highcharts（满足时间窗口滑动/十字光标/缩放）；ECharts 可选
- 编辑器：起步使用普通输入框；Monaco Editor 作为后续增强（可选）
- 样式：CSS Modules 或 Less/Sass（暂不使用 Tailwind）

后端：
- FastAPI + Uvicorn + Pydantic v2（简洁稳定）
- 认证：简易 JWT + SQLite 用户表（开发期可用本地用户配置）
- 任务执行：同步计算或 FastAPI BackgroundTasks；不使用 Celery/RQ/Redis
- 调度：通过命令行脚本或系统 cron 运行抓数；不使用 APScheduler
- 推送：暂不启用 WebSocket/SSE；结果同步返回或通过轮询接口获取

数据与计算：
- 数据源：CSV + SQLite 起步，使用 pandas（必要时再切换 Polars/DuckDB）
- 性能策略：chunk 分批读取、按日期/股票过滤、基础预聚合到本地 CSV/JSON
- 缓存：进程内 LRU/文件级缓存（不使用 Redis）

可观测性与安全：
- 日志：Python 标准 logging（模块/作业维度），简单日志文件持久化
- 输入校验：Pydantic；开启基本 CORS
- 资源保护：限制最大回测窗口与行数，避免长任务阻塞

测试与质量：
- 后端：pytest + coverage；ruff + black；mypy（可选）
- 前端：Vitest + Testing Library（后续补齐），先保障基本流程

部署与运行：
- 本地直接运行：`uvicorn app.main:app --reload` 与 `npm run dev`
- 生产：直接用 `uvicorn` +（可选）`nginx`；不使用 Docker/Compose
- 配置：`.env` 管理 Token/路径，提供 `.env.example`

交互与图表要求（细化）：
- 时间局部放大：区间选择（brush）、滚轮缩放、平移、十字光标、动态 tooltips（Highcharts 默认支持）
- 多曲线比较：策略净值与基准、分位净值 spread、滚动指标叠加
- 大数据优化：抽稀/窗口化；必要时后端下发预聚合序列（按月/周）
- 导出：图表快照与数据导出 CSV/JSON/PNG；可生成结果快照 ID

API 设计草案（简化）：
- POST `/auth/register`、`/auth/login`、`/auth/logout`（JWT，可后置）
- POST `/factor/preview`（DSL 安全解析与校验，返回表达式树/错误）
- POST `/backtest/run`（提交参数：表达式、股票池、时间窗、初始资金、约束；直接返回结果）
- GET `/backtest/result/{snapshot_id}`（可选：对历史结果的读取）

推荐最小组合（先跑通）：
- 前端：React + TypeScript + Vite + Ant Design + Highcharts（React Query 可选）
- 后端：FastAPI + Uvicorn + Pydantic v2 + SQLite（无 Redis/Celery）
- 数据：CSV + pandas；必要时再引入 DuckDB/Polars
- 质量：pytest/ruff/black；不使用 Docker/Sentry/Prometheus

里程碑拆分（精简执行路线）：
1) 数据接入：CSV→pandas 基本查询；（可选）SQLite 落库与简单索引
2) 基础回测：表达式在受控安全环境中执行，输出 PnL/Drawdown/IC/滚动指标
3) 前端界面：登录（可后置）、表达式输入、参数表单、结果框（图表+指标表）
4) 图表交互：Highcharts 接入，缩放/十字光标/热力图/分布图/分位收益
5) 结果管理：生成 snapshot_id，后端保存 JSON；前端可导出 CSV/PNG
6) 测试与上线：基础测试与格式化，`uvicorn` 直接部署（必要时 `nginx`）