# SCUTQUANT 项目说明

本说明面向本仓库的开发与使用，包含启动指南与详细项目结构，强调相对导入与相对路径的约定，避免硬编码的绝对路径。

## 启动指南

- 后端（FastAPI）
  - 在项目根目录运行：`python -m uvicorn backend.app.main:app --reload --port 8000`
  - 健康检查：打开 `http://127.0.0.1:8000/health`
  - 接口前缀：`/api/v1`（例如 `POST /api/v1/backtest/run`）

- 前端（React + Vite）
  - 推荐使用 `frontend-react`：
    - 进入目录：`cd frontend-react`
    - 启动：`npm run dev`（默认端口 `5174`）
    - 访问：`http://127.0.0.1:5174/`
  - 备用前端 `frontend`（实验用）：`cd frontend && npm run dev`（端口通常为 `5176`）

提示：命令行请使用 `python`，不要使用 `python3`。

## 项目结构

- 项目根：`/`（工作目录建议设为项目根）
  - `operators.py`：算子库（时间序列/截面操作，表达式构建的核心）
  - `dataset/`：数据目录（默认需包含 `all_historical_daily_data.csv`）
  - `backend/`：后端服务（FastAPI）
    - `app/main.py`：入口，注册路由与中间件
    - `app/api/v1/`：API 路由（`factor.py`、`backtest.py`）
    - `app/services/`：服务层（表达式引擎、回测引擎等）
  - `frontend-react/`：现代前端（React + Vite）
    - `src/`：页面与组件（暗色主题、结果图表）
    - `package.json`：脚本与依赖（`dev`、`build`）
  - `frontend/`：备用前端（精简）
  - `requirement.md`：需求总纲与技术栈
  - `project_structure_draft.md`：结构草案与运行说明（参考）
  - `log.md`：开发日志（可选）
  - `stage1/`：历史与对照代码（请不要修改此目录的内容）

## 数据

- 默认数据文件：`dataset/all_historical_daily_data.csv`
  - 最少字段：`ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount`
  - 后端会在缺少 `ret` 时按 `close/pre_close - 1` 自动构造日收益率
- 抓数脚本：`stage1/scripts/fetch_tushare_daily_all.py`
  - 示例：`python stage1/scripts/fetch_tushare_daily_all.py --start 20100101 --end 20251101 --out dataset/all_historical_daily_data.csv`

## 后端

- 入口：`backend/app/main.py`
  - CORS 全开（开发时便于前端请求）
  - 路由前缀：`/api/v1`
- 表达式相关：`backend/app/api/v1/factor.py`
  - `GET /api/v1/factor/operators`：返回可用算子列表（源自 `operators.py`）
- 回测接口：`backend/app/api/v1/backtest.py`
  - `POST /api/v1/backtest/run`：提交表达式与参数，返回净值、回撤、分布与指标
  - 参数示例：
    - `expr`：如 `cs_zscore(ts_rank(ts_corr(close, vol, 60), 15))`
    - `dataset_path`：默认 `dataset/all_historical_daily_data.csv`
    - `position_mode`：`long_only | short_only | long_short`
    - `t_plus`：`t0 | t1`
    - `start_date / end_date`：`YYYYMMDD`
    - `codes`：代码列表（可选）
    - `max_weight_per_stock`：单股权重上限（可选）

## 表达式引擎

- 文件：`backend/app/services/expression_engine.py`
- 加载算子库（相对导入）：
  - 优先通过 `import operators` 直接导入（项目根需在 `sys.path`，通常当前工作目录即项目根）
  - 若失败，回退为相对路径扫描 `<repo_root>/operators.py` 并加载，不再使用任何绝对路径
- 语法支持：
  - 允许：算术、括号、常量、名称、受限函数调用（仅环境中暴露的名字）
  - 支持多语句与赋值（最后一条需为表达式输出）
  - 禁止：属性调用（如 `np.log`）、越权访问（`obj.attr`）
- 名称映射：
  - `df` / `data` 暴露为完整 DataFrame
  - 每个列名映射为“单列 DataFrame”，以兼容 `operators.py` 中的签名
  - 函数映射：暴露 `operators.py` 中所有可调用对象

## 回测流程（简化）

- 加载数据：`load_dataset(csv_path)`
  - 统一索引为 `MultiIndex(ts_code, trade_date)`
  - 构造缺失的 `ret`
- 构建信号：`evaluate_expression(df, expr)`
  - 输出应为 `Series` 或单列 `DataFrame`
- 权重与交易：`build_weights`、`portfolio_returns_with_lots`
  - 规则：买入按 100 股整数倍，卖出按整数股；成交价取前收（或由 `close/(1+ret)` 估算）
  - 组合开盘总市值由前收与持仓确定；当日收益仅由价格变动产生
- 指标计算：净值、回撤、累计波动率与累计夏普、月度热力图、直方图与分位收益差

## 路径策略（相对优先）

- 算子库：相对导入（`import operators`），回退为相对文件路径扫描；不使用绝对文件系统路径
- 数据文件：默认使用相对路径 `dataset/all_historical_daily_data.csv`；可在请求体中传入其他相对/绝对路径
- 工作目录：建议始终在项目根执行后端与脚本，确保相对导入与相对路径成立

## 前端（简要）

- 主要目录：`frontend-react/src`（页面、组件、图表封装）
- 请求服务：使用 `axios` 访问后端（`http://127.0.0.1:8000/api/v1/...`）
- UI 与图表：Ant Design（暗色主题）、Highcharts（时间窗口、缩放、十字光标）

## 开发提示

- 不修改 `stage1/`（历史与对照代码）
- 性能建议：优先使用向量化算子（例如已优化的 `ts_corr`），减少 `rolling.apply` 与笛卡尔积计算规模；按代码与时间窗口分片运行大任务
- 运行/测试：尽量在较小股票池与时间范围先验证逻辑，再扩展到全市场与全时段