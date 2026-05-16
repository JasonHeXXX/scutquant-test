SCUTQUANT 项目目录结构草案（精简版，对齐 requirement.md）

根目录：
scutquant/
├── backend/                     # 后端服务（FastAPI，精简依赖）
│   ├── app/
│   │   ├── main.py             # 入口（创建 FastAPI 实例、路由、CORS、中间件）
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── auth.py     # 注册/登录/刷新/登出
│   │   │   │   ├── backtest.py # 回测运行（同步返回或生成快照）
│   │   │   │   └── factor.py   # 表达式预检/解析
│   │   ├── services/           # 业务服务（回测引擎、指标计算）
│   │   ├── models/             # Pydantic 模型与响应
│   │   ├── core/               # 配置、安全、错误处理（精简）
│   │   ├── db/                 # SQLite 连接与简易查询
│   │   └── config.py           # 环境变量与设置（.env）
│   ├── tests/                  # 后端测试（pytest）
│   ├── requirements.txt        # 生产依赖
│   ├── requirements-dev.txt    # 开发/测试依赖
│   
├── frontend/                    # 前端（React + Vite，默认黑色为底色）
│   ├── src/
│   │   ├── pages/              # 登录、账户、回测主页面（暗色主题）
│   │   ├── components/         # 通用组件（卡片、模态、表单、弹出结果框）
│   │   ├── features/           # 业务模块（表达式终端、参数下拉面板、结果框）
│   │   ├── charts/             # 图表封装（Highcharts）
│   │   ├── state/              # （可选）轻量状态管理或仅用组件状态
│   │   ├── services/           # API 客户端（axios），React Query hooks（可选）
│   │   ├── routes/             # 路由定义
│   │   ├── theme/              # 主题与暗色模式（黑色为底色，股票软件风格交互）
│   │   └── index.tsx
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── .eslintrc.cjs
│   ├── .prettierrc
│   
├── data/                        # 数据仓库（原始为主）
│   └── raw/                    # 原始文件（CSV 等）
├── scripts/                     # 抓数/导出/维护
│   ├── fetch_tushare_daily_all.py
│   └── export_pnl_json.py
├── docs/                        # 文档与规范
│   ├── frontend_dependencies.md
│   ├── backend_dependencies.md
│   └── project_structure_draft.md
└── .env.example                 # 环境变量示例（TOKEN、DB、REDIS 等）

迁移与兼容：
- 直接复用现有 `operators.py` 与脚本；`stage1/` 为历史记录，无需迁移。
- `dataset/all_historical_daily_data.csv` 作为原始入口；如需提速再评估 Parquet/DuckDB。

前端界面与交互（补充说明，落实 requirement.md 第 20–28 行）：
- 表达式输入终端框：命令行样式输入栏（起步可用 `<textarea>`，后续替换为 Monaco Editor），支持输入 operators.py 的算子组合；可直接使用 `open, high, low, close, pre_close, change, pct_chg, vol, amount` 字段作为参数。
- 参数下拉面板：
  - 股票池/代码选择（下拉+可输入）、时间窗口（开始/结束日期）、初始资金、单支股票占比上限；
  - 交易规则选择：`T0/T1`；持仓模式：`只做多/只做空/多空均可`。
- 结果展示框：
  - 弹出覆盖在表达式输入终端框右半部分；回测完成后自动弹出，或提前勾选常显；
  - 图表选项（7–8种，可切换）：累计净值、回撤、月度收益热力图、收益分布、滚动 Sharpe、滚动波动率、IC 与累计 IC、分位收益（多空 spread）。
  - 指标表（10–15项）：年化收益、年化波动率、Sharpe、Sortino、最大回撤、Calmar、胜率、盈亏比、平均日收益、偏度/峰度、VaR/ES、IR、日均换手、平均持有期、成本净效应（默认成本）。
- 图表交互风格：股票交易软件常见交互（时间窗口滑动、滚轮缩放、平移、十字光标、动态 tooltips）；不使用 matplotlib 样式。

API 与数据约定（精简后端）：
- POST `/factor/preview`：表达式预检（安全环境解析），返回 tokens/错误信息；
- POST `/backtest/run`：提交表达式与参数，返回 PnL/回撤/IC/指标表；
- 数据字段：CSV/SQLite 至少包含 `ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount`。

主题与样式：
- 全站默认暗色主题（黑色为底色），Ant Design 暗色主题切换，Highcharts 暗色主题配置；
- 关键组件（表达式输入、参数面板、结果弹窗、图表容器）采用暗色配色，适配日常交易软件的视觉习惯。

开发与运行（草案）：
- 前端：npm create vite@latest，安装依赖后本地运行（`npm run dev`）
- 后端：`uvicorn backend.app.main:app --reload` 直接启动；结果快照存本地 JSON/CSV

起步替代方案（快速预览）：
- 在 `frontend/` 先放置静态 `index.html + app.js + style.css`（暗色主题），包含表达式输入与参数下拉，按钮触发后端；待前端依赖准备好后再迁移到 Vite/AntD。