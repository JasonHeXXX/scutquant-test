import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
import functools
import inspect
import numbers


"""

与qlib的将数据简单加工后扔给ai模型找规律的思路不同, worldquant的思路是用精细的operators挖到有逻辑, 且回测表现良好的因子, 
用因子值构造中性组合, 直接用于投资
即qlib的量化投资流程是: 数据 -> 因子(这里的因子更接近feature的概念) -> 模型 -> 策略 -> 收益
而worldquant的流程是: 数据 -> 因子 -> 收益, 策略就是根据因子值构建投资组合(参考scutquant.alpha.market_neutralize的注释)
其实可以将qlib中的模型预测值当作worldquant中的因子, 那么qlib其实用是一种单因子策略进行投资, 只不过因子是由ai模型挖掘的, 而且策略更加多样
而worldquant的每一个因子都代表某个策略, 一个portfolio manager会选择多个因子, 并分配不同资金给每一个因子, 最后所有因子收益加总得到portfolio
一言以蔽之, worldquant 模式是量化1.0时代的经典模式, 而qlib的模式则适用于量化2.0甚至3.0时代. 
但这并不意味着两者是不兼容的. 事实上, 一个被精细加工过的feature能让模型的预测效果更好, 反过来模型的预测值也能作为一个很好的因子素材

scutquant的alpha模块用的是qlib的思路, 而为了让用户按照worldquant的方式构造自己的因子, 本模块应运而生
在本模块中, ts是对每个instrument在时序上计算, 而cs是在截面上计算, 所有返回结果都是pd.DataFrame
该模块提供了更加丰富的算子, 且速度也在不断优化. 计划以后alpha只提供因子表达式, 而具体计算由operators的算子完成
未来这部分可能会合并到alpha模块中, 让整个架构看起来不那么臃肿, 但也要考虑到合并后是否方便维护的问题

example:  

from operators import * 

factor = cs_zscore(ts_rank(ts_corr(df["close"], df["volume"], 15), 15))

"""




def multi_period_support(prefix_auto=True):
    """
    装饰器：为函数添加多周期支持
    
    自动检测所有数组类型参数，当发现列表/数组参数时，自动执行多周期计算并返回DataFrame
    支持多个数组参数同时存在，会生成所有可能的参数组合（笛卡尔积）
    
    Args:
        prefix_auto: 是否自动生成列名前缀，默认True
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 检查所有参数，找出数组类型的参数
            array_params = {}
            scalar_params = {}
            
            for param_name, param_value in bound_args.arguments.items():
                if isinstance(param_value, (list, tuple, np.ndarray)):
                    array_params[param_name] = param_value
                else:
                    scalar_params[param_name] = param_value
            
            # 如果有数组参数，执行多周期计算
            if array_params:
                import itertools
                
                func_name = func.__name__
                
                # 生成所有可能的参数组合（笛卡尔积）
                param_names = list(array_params.keys())
                param_values = list(array_params.values())
                combinations = list(itertools.product(*param_values))
                
                # 收集所有结果为列并按列拼接；不再自动重命名
                frames: list[pd.DataFrame] = []
                for combination in combinations:
                    # 创建当前组合的参数字典
                    current_args = scalar_params.copy()
                    # 添加当前组合的数组参数值
                    for i, param_name in enumerate(param_names):
                        current_args[param_name] = combination[i]
                    # 重新绑定参数并调用函数
                    new_bound_args = sig.bind(**current_args)
                    result = func(*new_bound_args.args, **new_bound_args.kwargs)
                    # 统一为 DataFrame
                    if isinstance(result, pd.Series):
                        # 若函数返回 Series，则尽量保留其 name；无 name 时仅在必要时使用表达式名
                        series_name = result.name if result.name else None
                        df_res = result.to_frame(name=series_name)
                    elif isinstance(result, pd.DataFrame):
                        df_res = result
                    else:
                        # 标量或其他类型，提升为单列 DataFrame；命名为函数名
                        df_res = pd.DataFrame(result, columns=[func_name]) if hasattr(result, '__len__') else pd.DataFrame({func_name: [result]})
                    frames.append(df_res)
                # 横向拼接所有周期结果，索引自动对齐
                if len(frames) == 1:
                    return frames[0]
                return pd.concat(frames, axis=1)
            
            # 正常执行原函数（没有数组参数）
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def df_scalar_broadcast_support(enable_inf_mask: bool = True,
                                mode: str = "both",
                                rename: bool = False,
                                prefix: str = "inf_mask"):
    """
    装饰器：允许在原本要求 DataFrame/DataFrameGroupBy 的参数位置输入标量（int/float/np.number），并按 DataFrame 形状进行广播。

    同时可选在返回值上进行无穷值屏蔽（inf/-inf → NaN）。

    支持两类广播情况：
    1) 至少一个 df-like 参数为 DataFrame/GroupBy：以第一个 df-like 的索引与列作为广播参考；
    2) 所有 df-like 参数均为标量：构造一个最小 MultiIndex（单日期、单资产）进行广播，保证函数能正常运行。

    返回值屏蔽参数：
    - enable_inf_mask: 是否对返回值进行 inf 屏蔽，默认 False 不改变现有行为；
    - mode: "both" | "pos" | "neg"，控制屏蔽正无穷、负无穷或两者；
    - rename: 是否重命名返回的列名（DataFrame/Series），添加前缀；
    - prefix: 列名前缀，默认 "inf_mask"。

    说明：
    - 仅处理类型注解为 `pd.DataFrame` 或 `pd.core.groupby.DataFrameGroupBy` 的参数；其它参数（如 n_period、rank 等）保持不变。
    - 与 multi_period_support 可叠加使用（建议将本装饰器放在最外层，使广播在多周期展开之前完成）。
    - 为了最大兼容已有配对逻辑（如按列名配对），当参考 DataFrame 存在时，广播出的常数列使用参考的列名。
    """


    def _is_scalar(x):
        return isinstance(x, numbers.Number) or isinstance(x, np.generic)

    def _is_df_like_annotation(ann) -> bool:
        # 统一用字符串判断，兼容 Union/| 的注解写法
        try:
            s = str(ann)
        except Exception:
            return False
        return ("DataFrame" in s) or ("DataFrameGroupBy" in s)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # 辅助函数：在 GroupBy 输入时严格尊重被选择的列
            def _extract_selection_df(gb: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
                # 优先使用 _selection（若存在且非空）
                try:
                    if hasattr(gb, '_selection') and gb._selection is not None:
                        sel = gb._selection
                        if isinstance(sel, list):
                            return gb.obj[sel]
                        else:
                            # 单列选择，确保返回 DataFrame
                            if sel in gb.obj.columns:
                                return gb.obj[[sel]]
                except Exception:
                    pass
                # 其次使用 _selected_obj（pandas 内部选择对象）
                try:
                    if hasattr(gb, '_selected_obj'):
                        selected = gb._selected_obj
                        if isinstance(selected, pd.DataFrame):
                            return selected
                        if isinstance(selected, pd.Series):
                            return selected.to_frame()
                except Exception:
                    pass
                # 再次尝试使用 _obj_with_exclusions（去除排除列后的对象）
                try:
                    if hasattr(gb, '_obj_with_exclusions'):
                        oe = gb._obj_with_exclusions
                        if isinstance(oe, pd.DataFrame):
                            return oe
                        if isinstance(oe, pd.Series):
                            return oe.to_frame()
                except Exception:
                    pass
                # 回退到完整的底层 DataFrame
                return gb.obj

            # 找到需要 DataFrame/DataFrameGroupBy 的参数名（基于注解）
            df_param_names: list[str] = []
            for pname, p in sig.parameters.items():
                ann = p.annotation
                if ann is not inspect._empty and _is_df_like_annotation(ann):
                    df_param_names.append(pname)

            # 选择参考 DataFrame（用于广播形状）
            ref_df: pd.DataFrame | None = None
            for pname in df_param_names:
                if pname in bound.arguments:
                    val = bound.arguments[pname]
                    if isinstance(val, pd.core.groupby.DataFrameGroupBy):
                        # 使用选择感知的 DataFrame 作为参考
                        ref_df = _extract_selection_df(val)
                        break
                    if isinstance(val, pd.DataFrame):
                        ref_df = val
                        break

            # 对标量输入进行广播；GroupBy 统一降为 DataFrame
            for pname in df_param_names:
                if pname not in bound.arguments:
                    continue
                val = bound.arguments[pname]

                if isinstance(val, pd.core.groupby.DataFrameGroupBy):
                    # 仅保留 GroupBy 中被选择的列，而不是完整 obj
                    bound.arguments[pname] = _extract_selection_df(val)
                    continue

                if _is_scalar(val):
                    scalar_value = float(val) if isinstance(val, (int, np.integer)) else float(val)
                    if ref_df is not None:
                        # 使用参考 DataFrame 的索引与列进行广播；保持列名以兼容现有按列名配对逻辑
                        bdf = pd.DataFrame(
                            np.full((len(ref_df.index), len(ref_df.columns)), scalar_value, dtype=float),
                            index=ref_df.index,
                            columns=list(ref_df.columns),
                        )
                    else:
                        # 构造最小 MultiIndex（单日期、单资产），保证函数可运行
                        idx = pd.MultiIndex.from_product(
                            [pd.to_datetime(["1970-01-01"]), ["A"]], names=["date", "asset"]
                        )
                        bdf = pd.DataFrame([scalar_value], index=idx, columns=[pname])
                    # 标记广播来源（目前仅作为元数据，不参与命名）
                    bdf.attrs["__broadcast_scalar__"] = {"param": pname, "value": scalar_value}
                    bound.arguments[pname] = bdf

            # 调用原函数
            result = func(*bound.args, **bound.kwargs)

            # 可选：对返回值进行 inf/-inf → NaN 屏蔽
            if enable_inf_mask:
                assert mode in ("both", "pos", "neg"), "mode 必须为 'both'|'pos'|'neg'"

                def _replace_inf_df(df: pd.DataFrame):
                    if mode == "pos":
                        df = df.replace(np.inf, np.nan)
                    elif mode == "neg":
                        df = df.replace(-np.inf, np.nan)
                    else:
                        df = df.replace([np.inf, -np.inf], np.nan)
                    if rename and hasattr(df, "columns"):
                        df.columns = [f"{prefix}({c})" for c in df.columns]
                    return df

                if isinstance(result, pd.DataFrame):
                    return _replace_inf_df(result)

                if isinstance(result, pd.Series):
                    if mode == "pos":
                        s = result.replace(np.inf, np.nan)
                    elif mode == "neg":
                        s = result.replace(-np.inf, np.nan)
                    else:
                        s = result.replace([np.inf, -np.inf], np.nan)
                    if rename and s.name:
                        s.name = f"{prefix}({s.name})"
                    return s

                if isinstance(result, pd.core.groupby.DataFrameGroupBy):
                    return result.apply(lambda df: _replace_inf_df(df))

                if isinstance(result, np.ndarray):
                    arr = result.copy()
                    if mode == "pos":
                        arr[np.isposinf(arr)] = np.nan
                    elif mode == "neg":
                        arr[np.isneginf(arr)] = np.nan
                    else:
                        arr[np.isinf(arr)] = np.nan
                    return arr

                if isinstance(result, numbers.Number) or isinstance(result, np.generic):
                    if (mode in ("both", "pos") and result == np.inf) or (mode in ("both", "neg") and result == -np.inf):
                        return np.nan
                    return result

                return result

            return result

        return wrapper

    return decorator

 
@df_scalar_broadcast_support()
def demean(data:  pd.DataFrame | pd.core.groupby.DataFrameGroupBy) ->  pd.DataFrame:
    """
    Cross-sectional demean per date: x - mean(x) across instruments.
    - Supports DataFrame and DataFrameGroupBy inputs
    - Returns DataFrame with columns: demean(col)
    - No Series outputs; preserves original index
    """
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: x - x.mean())
        res.columns = [f"demean({col})" for col in data.columns]
        return res
    # GroupBy input
    res = data.transform(lambda x: x - x.mean())
    # Determine original columns
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"demean({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"demean({col})" for col in original_columns]
        else:
            res.columns = [f"demean({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def mean(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise mean of two inputs with safe column pairing and standardized naming.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: mean(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("mean only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = ((data1[c1] + data2[c2]) / 2)
        colname = f"mean({c1}, {c2})"
        result_frames.append(res_series.to_frame(name=colname))

    result = pd.concat(result_frames, axis=1)
    return result


@df_scalar_broadcast_support()
def sign(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise sign operator for DataFrame and DataFrameGroupBy inputs.
    - Ensures the return type is always DataFrame (no Series outputs)
    - Preserves original index and avoids forcing index names
    - Standardizes column naming to: sign(<original_col_name>)
    - Supports single/multiple column selections for GroupBy
    """
    # DataFrame input: vectorized sign and standardized column naming
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(np.sign(data.values), index=data.index)
        res.columns = [f"sign({col})" for col in data.columns]
        return res

    # DataFrameGroupBy input: use transform to apply sign per group
    res = data.transform(lambda x: np.sign(x))

    # Determine original column names (respect groupby selection if present)
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']

    # Ensure DataFrame output and set standardized column names
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"sign({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"sign({col})" for col in original_columns]
        else:
            # Fallback to current columns to avoid mismatch
            res.columns = [f"sign({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def abs(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise absolute value for DataFrame and DataFrameGroupBy inputs.
    - Ensures the return type is always DataFrame (no Series outputs)
    - Preserves original index and avoids forcing index names
    - Standardizes column naming to: abs(<original_col_name>)
    - Supports single/multiple column selections for GroupBy
    """
    # DataFrame input: vectorized abs and standardized column naming
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(np.abs(data.values), index=data.index)
        res.columns = [f"abs({col})" for col in data.columns]
        return res

    # DataFrameGroupBy input: use transform to apply abs per group
    res = data.transform(lambda x: np.abs(x))

    # Determine original column names (respect groupby selection if present)
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']

    # Ensure DataFrame output and set standardized column names
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"abs({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"abs({col})" for col in original_columns]
        else:
            # Fallback to current columns to avoid mismatch
            res.columns = [f"abs({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def sqrt(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise square root for DataFrame and DataFrameGroupBy inputs.
    - Ensures the return type is always DataFrame (no Series outputs)
    - Preserves original index and avoids forcing index names
    - Standardizes column naming to: sqrt(<original_col_name>)
    - Supports single/multiple column selections for GroupBy
    - Negative values are converted to NaN (mathematically correct behavior)
    """
    # DataFrame input: vectorized sqrt and standardized column naming
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(np.sqrt(data.values), index=data.index)
        res.columns = [f"sqrt({col})" for col in data.columns]
        return res

    # DataFrameGroupBy input: use transform to apply sqrt per group
    res = data.transform(lambda x: np.sqrt(x))

    # Determine original column names (respect groupby selection if present)
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']

    # Ensure DataFrame output and set standardized column names
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"sqrt({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"sqrt({col})" for col in original_columns]
        else:
            # Fallback to current columns to avoid mismatch
            res.columns = [f"sqrt({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def add(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
        data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise addition of two inputs with safe column pairing and standardized naming.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: add(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("add only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] + data2[c2]
        colname = f"{c1}+{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def subtract(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
             data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise subtraction of two inputs with safe column pairing and standardized naming.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: subtract(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("subtract only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] - data2[c2]
        colname = f"{c1}-{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def multiply(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
             data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise multiplication of two inputs with safe column pairing and standardized naming.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: multiply(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("multiply only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] * data2[c2]
        colname = f"{c1}*{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def divide(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
           data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise division of two inputs with safe column pairing and standardized naming.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: divide(col1, col2)
    - Preserves original index and does not force index names
    - Division by zero results in inf/NaN as per numpy behavior
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("divide only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] / data2[c2]
        colname = f"{c1}/{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def sign_power(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, p: float) -> pd.DataFrame:
    """
    Element-wise signed power: sign(x) * |x|^p
    - Supports DataFrame and DataFrameGroupBy inputs; Series not allowed
    - Column names: sign_power(col, p)
    """
    if isinstance(data, pd.DataFrame):
        s = np.sign(data.values)
        mag = np.power(np.abs(data.values), p)
        res = pd.DataFrame(s * mag, index=data.index)
        res.columns = [f"sign_power({col}, {p})" for col in data.columns]
        return res
    # GroupBy input
    res = data.transform(lambda x: np.sign(x) * (np.abs(x) ** p))
    # Determine original columns
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"sign_power({col_name}, {p})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"sign_power({col}, {p})" for col in original_columns]
        else:
            res.columns = [f"sign_power({col}, {p})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def log(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise natural logarithm with standardized column naming.
    - Supports DataFrame and DataFrameGroupBy inputs; Series not allowed
    - Column names: log(col)
    - Does not modify index names
    """
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(np.log(data.values), index=data.index)
        res.columns = [f"log({col})" for col in data.columns]
        return res
    # GroupBy input
    res = data.transform(lambda x: np.log(x))
    # Determine original columns
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']
    # Ensure DataFrame output and set names
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"log({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"log({col})" for col in original_columns]
        else:
            res.columns = [f"log({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def tanh(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise hyperbolic tangent with standardized column naming.
    - Supports DataFrame and DataFrameGroupBy inputs; Series not allowed
    - Column names: tanh(col)
    """
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(np.tanh(data.values), index=data.index)
        res.columns = [f"tanh({col})" for col in data.columns]
        return res
    res = data.transform(lambda x: np.tanh(x))
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"tanh({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"tanh({col})" for col in original_columns]
        else:
            res.columns = [f"tanh({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def sigmoid(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise logistic sigmoid with standardized column naming.
    - Supports DataFrame and DataFrameGroupBy inputs; Series not allowed
    - Column names: sigmoid(col)
    """
    if isinstance(data, pd.DataFrame):
        res = pd.DataFrame(1 / (1 + np.exp(-data.values)), index=data.index)
        res.columns = [f"sigmoid({col})" for col in data.columns]
        return res
    res = data.transform(lambda x: 1 / (1 + np.exp(-x)))
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']
    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"sigmoid({col_name})")
    else:
        if len(res.columns) == len(original_columns):
            res.columns = [f"sigmoid({col})" for col in original_columns]
        else:
            res.columns = [f"sigmoid({col})" for col in list(res.columns)]
    return res


@df_scalar_broadcast_support()
def bigger(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise max between two inputs with safe column pairing.
    - Accepts DataFrame or DataFrameGroupBy; Series not allowed
    - Pairs columns by common names; defaults to first columns
    - Column names: bigger(col1, col2)
    """
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("bigger only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        series = data1[c1].where(data1[c1] > data2[c2], data2[c2])
        result_frames.append(series.to_frame(name=f"bigger({c1}, {c2})"))

    return pd.concat(result_frames, axis=1)


@multi_period_support()
def smaller(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise min between two inputs with safe column pairing.
    - Accepts DataFrame or DataFrameGroupBy; Series not allowed
    - Pairs columns by common names; defaults to first columns
    - Column names: smaller(col1, col2)
    """
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("smaller only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        series = data1[c1].where(data1[c1] < data2[c2], data2[c2])
        result_frames.append(series.to_frame(name=f"smaller({c1}, {c2})"))

    return pd.concat(result_frames, axis=1)

@df_scalar_broadcast_support()
def power(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
          data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise power operation (data1 ** data2) with safe column pairing.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: power(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("power only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] ** data2[c2]
        colname = f"{c1}**{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def floor_divide(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
                 data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise floor division (data1 // data2) with safe column pairing.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: floor_divide(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("floor_divide only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] // data2[c2]
        colname = f"{c1}//{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def modulo(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, 
           data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Element-wise modulo operation (data1 % data2) with safe column pairing.
    - Accepts DataFrame or DataFrameGroupBy; Series is not allowed
    - Pairs columns by common names; if none, pairs the first columns
    - Returns DataFrame with columns named: modulo(col1, col2)
    - Preserves original index and does not force index names
    """
    # Normalize GroupBy to underlying DataFrame
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj
    # Strictly disallow Series
    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("modulo only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # Align indexes automatically via arithmetic; ensure DataFrame output
        res_series = data1[c1] % data2[c2]
        colname = f"{c1}%{c2}"
        result_frames.append(res_series.to_frame(name=colname))

    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def and_(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
         data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    逻辑与（AND）：对两个输入逐元素做逻辑与，支持 DataFrame/GroupBy。
    - 非零且非 NaN 视为 True
    - 返回布尔 DataFrame，列名：<c1>&<c2>
    - 列配对：按同名列配对；若无同名列，使用各自第一列
    """
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj

    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("and_ 仅支持 DataFrame 或 DataFrameGroupBy 输入")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        l = data1[c1]
        r = data2[c2]
        res_series = (l.notna() & (l != 0)) & (r.notna() & (r != 0))
        result_frames.append(res_series.to_frame(name=f"{c1}&{c2}"))
    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def or_(data1: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
        data2: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    逻辑或（OR）：对两个输入逐元素做逻辑或，支持 DataFrame/GroupBy。
    - 非零且非 NaN 视为 True
    - 返回布尔 DataFrame，列名：<c1>|<c2>
    - 列配对：按同名列配对；若无同名列，使用各自第一列
    """
    if isinstance(data1, pd.core.groupby.DataFrameGroupBy):
        data1 = data1.obj
    if isinstance(data2, pd.core.groupby.DataFrameGroupBy):
        data2 = data2.obj

    if isinstance(data1, pd.Series) or isinstance(data2, pd.Series):
        raise TypeError("or_ 仅支持 DataFrame 或 DataFrameGroupBy 输入")

    cols1 = list(data1.columns)
    cols2 = list(data2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        l = data1[c1]
        r = data2[c2]
        res_series = (l.notna() & (l != 0)) | (r.notna() & (r != 0))
        result_frames.append(res_series.to_frame(name=f"{c1}|{c2}"))
    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def if_else(cond: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
            a: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
            b: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    条件选择（if_else）：若 cond 为真则选 a，否则选 b，支持 DataFrame/GroupBy。
    - 逻辑判定：非零且非 NaN 视为 True，等于 0 或 NaN 视为 False
    - 列配对：优先按同名列在三者中同时存在进行配对；若不存在，则以 cond 的每列与 a、b 的第一列配对
    - 列名规范：if_else(<cond_col>, <a_col>, <b_col>)
    """
    # 统一 GroupBy -> DataFrame
    cond_df = cond.obj if isinstance(cond, pd.core.groupby.DataFrameGroupBy) else cond
    a_df = a.obj if isinstance(a, pd.core.groupby.DataFrameGroupBy) else a
    b_df = b.obj if isinstance(b, pd.core.groupby.DataFrameGroupBy) else b

    # 严格禁止 Series（保持与 and_/or_ 一致的输入约束）
    if isinstance(cond_df, pd.Series) or isinstance(a_df, pd.Series) or isinstance(b_df, pd.Series):
        raise TypeError("if_else 仅支持 DataFrame 或 DataFrameGroupBy 输入")

    cond_cols = list(cond_df.columns)
    a_cols = list(a_df.columns)
    b_cols = list(b_df.columns)

    # 三者公共列优先配对；否则使用 a、b 的第一列
    common = [c for c in cond_cols if (c in a_cols and c in b_cols)]
    triples = [(c, c, c) for c in common] if len(common) > 0 else [(c, a_cols[0], b_cols[0]) for c in cond_cols]

    result_frames = []
    for c_cond, c_a, c_b in triples:
        mask = (cond_df[c_cond].notna() & (cond_df[c_cond] != 0))
        res_series = a_df[c_a].where(mask, b_df[c_b])
        result_frames.append(res_series.to_frame(name=f"if_else({c_cond}, {c_a}, {c_b})"))
    return pd.concat(result_frames, axis=1)


@df_scalar_broadcast_support()
def is_nan(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    NaN 检测（is_nan）：将 NaN 标记为 1，非 NaN 标记为 0，支持 DataFrame/GroupBy。
    - 输出为 0/1（整数型）
    - 列名规范：is_nan(<col>)
    """
    # DataFrame 直接处理
    if isinstance(data, pd.DataFrame):
        res = data.isna().astype(int)
        res.columns = [f"is_nan({c})" for c in data.columns]
        return res

    # GroupBy：逐列 transform 并恢复列名
    res = data.transform(lambda x: x.isna().astype(int))

    # 恢复原始列名（参考 not_ 中的命名策略）
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']

    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"is_nan({col_name})")
    else:
        cols = list(res.columns)
        if len(cols) == len(original_columns):
            res.columns = [f"is_nan({c})" for c in original_columns]
        else:
            res.columns = [f"is_nan({c})" for c in cols]
    return res


"""
MAD缩尾算子 - 使用中位数绝对偏差进行异常值处理
"""
@df_scalar_broadcast_support()
def mad_winsor(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    # 统一 DataFrame 输入（若为 GroupBy，回退到其原始 DataFrame）
    df = data.obj if isinstance(data, pd.core.groupby.DataFrameGroupBy) else data
    med = df.groupby(level=0).median()
    mad = (df - med).abs().groupby(level=0).median()
    up = med + 3 * mad * 1.4826
    down = med - 3 * mad * 1.4826
    result = df.clip(upper=up, lower=down)
    # 统一列名为函数(原列名, 参数)格式
    result.columns = [f"mad_winsor({col})" for col in df.columns]
    return result

"""
无穷值掩码算子 - 将无穷值替换为NaN
"""
@df_scalar_broadcast_support()
def inf_mask(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    Replace inf with nan
    """
    df = data.obj if isinstance(data, pd.core.groupby.DataFrameGroupBy) else data
    df = df.where(df != np.inf, np.nan)
    result = df.where(df != -np.inf, np.nan)
    # 统一列名为函数(原列名, 参数)格式
    result.columns = [f"inf_mask({col})" for col in df.columns]
    return result

@df_scalar_broadcast_support()
def neutralize(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
               target: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
               features: pd.DataFrame | pd.core.groupby.DataFrameGroupBy = None,
               n_jobs=-1) -> pd.DataFrame:
    """
    在截面上对选定的features进行target中性化, 剩余因子不变

    example:

    # 使用补充数据data, 对factor_raw的RSI, MACD和KDJ_K因子进行市值中性化

    factor_neutralized = alpha.neutralize(factor_raw, target=data["ln_market_value"], features=factor_raw[["RSI", "MACD", "KDJ_K"]])

    :param data: 需要中性化的因子集合
    :param target: 解释变量
    :param features: 需要中性化的因子DataFrame, 如果为None则对整个data进行中性化
    :param n_jobs: 同时调用的cpu数
    :return: pd.DataFrame, 包括中性化后的因子和未中性化的其它因子
    """
    # 统一 DataFrame 输入（允许 Series 作为中间变量，立即转为 DataFrame）
    df = data.obj if isinstance(data, pd.core.groupby.DataFrameGroupBy) else data
    tgt = target.obj if isinstance(target, pd.core.groupby.DataFrameGroupBy) else target
    fts = features.obj if isinstance(features, pd.core.groupby.DataFrameGroupBy) else features
    if isinstance(tgt, pd.Series):
        tgt = tgt.to_frame()
    if fts is not None and isinstance(fts, pd.Series):
        fts = fts.to_frame()
    # 统一索引对齐到主输入 df
    if not tgt.index.equals(df.index):
        tgt = tgt.reindex(df.index)
    if fts is not None and not fts.index.equals(df.index):
        fts = fts.reindex(df.index)

    if fts is None:
        # 如果没有指定features，对整个data进行中性化
        result = cs_resid(tgt, df)
        # 保证返回 DataFrame
        if isinstance(result, pd.Series):
            result = result.to_frame()
        # 重命名列为 neutralize(原列名, 目标列名, None, n_jobs)
        new_cols = []
        for col in list(result.columns):
            # 期望格式：cs_resid(tgt_col, df_col)
            if isinstance(col, str) and col.startswith("cs_resid(") and col.endswith(")") and "," in col:
                inner = col[len("cs_resid("):-1]
                parts = [p.strip() for p in inner.split(",")]
                if len(parts) == 2:
                    tgt_name, df_name = parts[0], parts[1]
                    new_cols.append(f"neutralize({df_name}, {tgt_name}, None, {n_jobs})")
                    continue
            # 回退：使用第一列目标名
            tgt_name_fallback = list(tgt.columns)[0] if hasattr(tgt, 'columns') and len(tgt.columns) > 0 else 'target'
            new_cols.append(f"neutralize({col}, {tgt_name_fallback}, None, {n_jobs})")
        result.columns = new_cols
        return result
    else:
        # 过滤不存在的列，避免列名不匹配错误
        feature_cols = [c for c in fts.columns.tolist() if c in df.columns]
        other_cols = [c for c in df.columns if c not in feature_cols]
        # 若过滤后为空，则直接返回原始数据
        if not feature_cols:
            # 没有需要中性化的列，也统一命名返回为 neutralize(col, tgt, None, n_jobs)
            tgt_name = list(tgt.columns)[0] if hasattr(tgt, 'columns') and len(tgt.columns) > 0 else 'target'
            res = df.copy()
            res.columns = [f"neutralize({col}, {tgt_name}, None, {n_jobs})" for col in res.columns]
            return res
        factor_neu = Parallel(n_jobs=n_jobs)(delayed(cs_resid)(tgt, fts[[f]]) for f in feature_cols)
        data_neu = pd.concat(factor_neu, axis=1)
        # 将中性化列命名为 neutralize(原列名, 目标列名, 特征列名, n_jobs)
        tgt_name = list(tgt.columns)[0] if hasattr(tgt, 'columns') and len(tgt.columns) > 0 else 'target'
        data_neu.columns = [f"neutralize({f}, {tgt_name}, {f}, {n_jobs})" for f in feature_cols]
        # 其余列保持数值不变，但统一命名为 neutralize(col, tgt, None, n_jobs)
        other_df = df[other_cols].copy()
        other_df.columns = [f"neutralize({c}, {tgt_name}, None, {n_jobs})" for c in other_cols]
        result = pd.concat([data_neu, other_df], axis=1)
        return result


@df_scalar_broadcast_support()
def factor_neutralize(factors: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
                      target: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
                      feature: pd.DataFrame | pd.core.groupby.DataFrameGroupBy = None) -> pd.DataFrame:
    # 直接按传参与列归属生成列名
    df = factors.obj if isinstance(factors, pd.core.groupby.DataFrameGroupBy) else factors
    tgt_df = target.obj if isinstance(target, pd.core.groupby.DataFrameGroupBy) else target
    feat_df = feature.obj if isinstance(feature, pd.core.groupby.DataFrameGroupBy) else feature
    tgt_name = list(tgt_df.columns)[0] if hasattr(tgt_df, 'columns') and len(tgt_df.columns) > 0 else 'target'
    if feat_df is None:
        # 不指定特征：对整表执行中性化，命名为 factor_neutralize(col, tgt, None)
        res = neutralize(df, features=None, target=tgt_df)
        res.columns = [f"factor_neutralize({c}, {tgt_name}, None)" for c in res.columns]
        return res
    # 指定特征：对每个特征列生成对应列名，其余列为 None
    feature_cols = [c for c in feat_df.columns.tolist() if c in df.columns]
    other_cols = [c for c in df.columns if c not in feature_cols]
    if not feature_cols:
        res = neutralize(df, features=None, target=tgt_df)
        res.columns = [f"factor_neutralize({c}, {tgt_name}, None)" for c in res.columns]
        return res
    factor_neu = Parallel(n_jobs=-1)(delayed(cs_resid)(tgt_df, feat_df[[f]]) for f in feature_cols)
    data_neu = pd.concat(factor_neu, axis=1)
    data_neu.columns = [f"factor_neutralize({f}, {tgt_name}, {f})" for f in feature_cols]
    other_df = df[other_cols].copy()
    other_df.columns = [f"factor_neutralize({c}, {tgt_name}, None)" for c in other_cols]
    return pd.concat([data_neu, other_df], axis=1)


@df_scalar_broadcast_support()
def market_neutralize(x: pd.DataFrame | pd.core.groupby.DataFrameGroupBy, long_only: bool = False) -> pd.DataFrame:
    """
    市场组合中性化:
    (1) 对所有股票减去其截面上的因子均值
    (2) 在(1)之后, 对每支股票除以截面上的因子值绝对值之和

    这样处理后每支股票会获得一个权重, 代表着资金的方向和数量(例如0.5代表半仓做多, -0.25代表1/4仓做空),
    且截面上的权重之和为0, 绝对值之和为1.
    """
    df = x.obj if isinstance(x, pd.core.groupby.DataFrameGroupBy) else x
    _mean = df.groupby(level=0).mean()
    df = df - _mean
    abs_sum = df.abs().groupby(level=0).sum()
    df = df / abs_sum
    if long_only:
        df[df < 0] = 0
        df = df * 2
    # 统一列名为函数(原列名, 参数)格式
    df.columns = [f"market_neutralize({col}, {long_only})" for col in df.columns]
    return df


"""
协方差算子 - 计算两个DataFrame的协方差
"""
#wfy
@df_scalar_broadcast_support()
def cov(x: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
        y: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    计算两个DataFrame的协方差
    
    Parameters:
    -----------
    x : pd.DataFrame
        第一个DataFrame
    y : pd.DataFrame  
        第二个DataFrame
        
    Returns:
    --------
    pd.DataFrame
        协方差结果DataFrame
    """
    # 统一 DataFrame 输入
    x_df = x.obj if isinstance(x, pd.core.groupby.DataFrameGroupBy) else x
    y_df = y.obj if isinstance(y, pd.core.groupby.DataFrameGroupBy) else y
    # 对齐索引
    if not x_df.index.equals(y_df.index):
        y_df = y_df.reindex(x_df.index)
    # 列配对：优先使用交集，否则配对各自第一列
    cols1 = list(x_df.columns)
    cols2 = list(y_df.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
    # 逐对计算整体样本协方差，并广播为常数列到索引
    result_frames: list[pd.DataFrame] = []
    for c1, c2 in pairs:
        df_pair = pd.concat([
            x_df[[c1]].rename(columns={c1: "x"}),
            y_df[[c2]].rename(columns={c2: "y"})
        ], axis=1)
        val = df_pair["x"].cov(df_pair["y"])  # 标量
        series = pd.Series(val, index=x_df.index)
        result_frames.append(series.to_frame(name=f"cov({c1}, {c2})"))
    result = pd.concat(result_frames, axis=1)
    return result

"""
皮尔逊相关系数算子 - 计算两个DataFrame的皮尔逊相关系数
"""
#wfy
@df_scalar_broadcast_support()
def pearson_corr(x: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
                 y: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    计算两个DataFrame的皮尔逊相关系数
    
    Parameters:
    -----------
    x : pd.DataFrame
        第一个DataFrame
    y : pd.DataFrame
        第二个DataFrame
        
    Returns:
    --------
    pd.DataFrame
        皮尔逊相关系数结果DataFrame
    """
    # 统一 DataFrame 输入
    x_df = x.obj if isinstance(x, pd.core.groupby.DataFrameGroupBy) else x
    y_df = y.obj if isinstance(y, pd.core.groupby.DataFrameGroupBy) else y
    # 对齐索引
    if not x_df.index.equals(y_df.index):
        y_df = y_df.reindex(x_df.index)
    # 列配对：优先使用交集，否则配对各自第一列
    cols1 = list(x_df.columns)
    cols2 = list(y_df.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
    # 逐对计算整体样本皮尔逊相关，并广播为常数列到索引
    result_frames: list[pd.DataFrame] = []
    for c1, c2 in pairs:
        df_pair = pd.concat([
            x_df[[c1]].rename(columns={c1: "x"}),
            y_df[[c2]].rename(columns={c2: "y"})
        ], axis=1)
        val = df_pair["x"].corr(df_pair["y"])  # 标量
        series = pd.Series(val, index=x_df.index)
        result_frames.append(series.to_frame(name=f"pearson_corr({c1}, {c2})"))
    result = pd.concat(result_frames, axis=1)
    return result

"""
互信息评分时序算子 - 计算特征与目标值的互信息
"""
#wfy
@df_scalar_broadcast_support()
def make_mi_scores(X: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
                   y: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    :param X: pd.DataFrame, 输入的特征
    :param y: pd.DataFrame, 输入的目标值
    :return: pd.DataFrame, index为特征名，value为mutual information
    """
    # 统一 DataFrame 输入
    X_df = X.obj if isinstance(X, pd.core.groupby.DataFrameGroupBy) else X
    y_df = y.obj if isinstance(y, pd.core.groupby.DataFrameGroupBy) else y
    # Label encoding for categoricals
    for colname in X_df.select_dtypes("object"):
        # factorize 返回 Series，用作中间变量，写回 DataFrame
        X_df[colname], _ = X_df[colname].factorize()
    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X_df.dtypes == int
    # 索引对齐
    if not y_df.index.equals(X_df.index):
        y_df = y_df.reindex(X_df.index)
    # y 必须为 1D，取第一列作为目标（中间阶段允许 Series）
    y_1d = y_df.iloc[:, 0]
    mi_scores = mutual_info_regression(X_df, y_1d, discrete_features=discrete_features)
    y_col = y_df.columns[0] if hasattr(y_df, 'columns') and len(y_df.columns) > 0 else 'y'
    # 获取 X 的名称，统一使用 "matrix"
    x_name = "matrix"
    # 列名应严格包含两个参数：X 与 y 列名
    mi_scores = pd.DataFrame(mi_scores, index=X_df.columns, columns=[f"make_mi_scores({x_name}, {y_col})"])
    mi_scores = mi_scores.sort_values(by=f"make_mi_scores({x_name}, {y_col})", ascending=False)
    return mi_scores

"""
相关系数评分时序算子 - 计算特征与目标值的相关系数
"""
#wfy
@df_scalar_broadcast_support()
def make_r_scores(X: pd.DataFrame | pd.core.groupby.DataFrameGroupBy,
                  y: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    :param X: pd.DataFrame, 特征值
    :param y: pd.DataFrame, 目标值
    :return: pd.DataFrame, index为特征名, value为相关系数
    """
    # 统一 DataFrame 输入
    X_df = X.obj if isinstance(X, pd.core.groupby.DataFrameGroupBy) else X
    y_df = y.obj if isinstance(y, pd.core.groupby.DataFrameGroupBy) else y
    # 索引对齐
    if not y_df.index.equals(X_df.index):
        y_df = y_df.reindex(X_df.index)
    # 选取 y 的第一列作为目标
    target_col = y_df.columns[0]
    scores: list[float] = []
    cols = list(X_df.columns)
    for c in cols:
        # 使用 DataFrame 切片避免 Series 输入
        corr_df = pearson_corr(X_df[[c]], y_df[[target_col]])
        # 从 DataFrame 中提取标量值（中间变量）
        val = float(corr_df.iloc[0, 0])
        scores.append(val)
    y_col = y_df.columns[0] if hasattr(y_df, 'columns') and len(y_df.columns) > 0 else 'y'
    # 获取 X 的名称，统一使用 "matrix"
    x_name = "matrix"
    # 列名应严格包含两个参数：X 与 y 列名
    result = pd.DataFrame({f"make_r_scores({x_name}, {y_col})": scores}, index=cols).sort_values(f"make_r_scores({x_name}, {y_col})", ascending=False)
    return result




def ts_backfill(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    时间序列后向填充函数
    
    参数:
    data: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，时间序列数据
    
    返回:
    pd.DataFrame，后向填充后的数据
    """
    if isinstance(data, pd.DataFrame):
        # 数据验证
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("data必须具有MultiIndex（datetime, instrument）")
        
        if len(data.index.names) != 2:
            raise ValueError("data的索引必须恰好有两个层级")
        
        result = data.groupby(level=1).transform(lambda x: x.bfill())
        # 修改列名格式
        original_columns = result.columns
        new_columns = [f"ts_backfill({col})" for col in original_columns]
        result.columns = new_columns
        return result
    else:
        res = data.transform(lambda x: x.bfill())
        # 修改列名格式
        original_columns = res.columns
        new_columns = [f"ts_backfill({col})" for col in original_columns]
        res.columns = new_columns
        return res



def ts_ffill(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    时间序列前向填充函数
    
    参数:
    data: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，时间序列数据
    
    返回:
    pd.DataFrame，前向填充后的数据
    """
    if isinstance(data, pd.DataFrame):
        # 数据验证
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("data必须具有MultiIndex（datetime, instrument）")
        
        if len(data.index.names) != 2:
            raise ValueError("data的索引必须恰好有两个层级")
        
        result = data.groupby(level=1).transform(lambda x: x.ffill())
        # 修改列名格式
        original_columns = result.columns
        new_columns = [f"ts_ffill({col})" for col in original_columns]
        result.columns = new_columns
        return result
    else:
        res = data.transform(lambda x: x.ffill())
        # 修改列名格式
        original_columns = res.columns
        new_columns = [f"ts_ffill({col})" for col in original_columns]
        res.columns = new_columns
        return res

"""
移动calc_autocorr函数到operators.py，理由：属于时间序列算子操作。
"""
#wfy
@multi_period_support()
@df_scalar_broadcast_support()
def ts_calc_autocorr(feature: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    计算金融工具的自相关系数（去除绘图功能的版本）
    
    参数:
    feature: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，具有datetime和instrument多重索引的时间序列数据
    lag: int，自相关的滞后期，默认为1
    
    返回:
    pd.DataFrame，每个金融工具的自相关系数
    """
    # 使用groupby优化的自相关计算
    def calc_autocorr_for_group(group):
        if len(group) <= lag:
            return np.nan
        # 处理单列和多列情况
        if isinstance(group, pd.DataFrame):
            # 对于多列DataFrame，计算每列的自相关
            result = {}
            for col in group.columns:
                try:
                    result[col] = group[col].autocorr(lag=lag)
                except Exception:
                    result[col] = np.nan
            return pd.Series(result)
        else:
            # 对于Series，直接计算自相关
            try:
                return group.autocorr(lag=lag)
            except Exception:
                return np.nan
    
    if isinstance(feature, pd.DataFrame):
        # 数据验证
        if not isinstance(feature.index, pd.MultiIndex):
            raise ValueError("feature必须具有MultiIndex（datetime, instrument）")
        
        if len(feature.index.names) != 2:
            raise ValueError("feature的索引必须恰好有两个层级")
        
        # 按instrument分组计算自相关
        autocorr_result = feature.groupby(level=1).apply(calc_autocorr_for_group)
        
        # 获取原始列名
        if len(feature.columns) == 1:
            feature_name = feature.columns[0]
        else:
            # 多列情况下，使用所有列名
            feature_name = ', '.join(feature.columns)
        
        # 确保结果是 DataFrame 格式并设置正确的列名
        if isinstance(autocorr_result, pd.Series):
            autocorr_result = autocorr_result.to_frame()
        
        # 修改列名格式
        if len(feature.columns) == 1:
            autocorr_result.columns = [f"ts_calc_autocorr({feature_name}, {lag})"]
        else:
            # 多列情况下，为每列生成对应的列名
            new_columns = []
            for col in feature.columns:
                new_columns.append(f"ts_calc_autocorr({col}, {lag})")
            autocorr_result.columns = new_columns
    else:
        # 直接在已分组的对象上应用自相关计算
        autocorr_result = feature.apply(calc_autocorr_for_group)
        
        # 获取原始列名（从 DataFrameGroupBy 对象中获取）
        if hasattr(feature, 'obj') and hasattr(feature.obj, 'columns'):
            if len(feature.obj.columns) == 1:
                feature_name = feature.obj.columns[0]
            else:
                # 多列情况下，使用所有列名
                feature_name = ', '.join(feature.obj.columns)
        else:
            feature_name = 'feature'
        
        # 确保结果是 DataFrame 格式并设置正确的列名
        if isinstance(autocorr_result, pd.Series):
            autocorr_result = autocorr_result.to_frame()
        
        # 修改列名格式
        if hasattr(feature, 'obj') and hasattr(feature.obj, 'columns') and len(feature.obj.columns) > 1:
            # 多列情况下，为每列生成对应的列名
            new_columns = []
            for col in feature.obj.columns:
                new_columns.append(f"ts_calc_autocorr({col}, {lag})")
            autocorr_result.columns = new_columns
        else:
            autocorr_result.columns = [f"ts_calc_autocorr({feature_name}, {lag})"]
    
    return autocorr_result

"""
价格转收益率时序算子 - 计算价格序列的收益率
"""
#wfy
@multi_period_support()
@df_scalar_broadcast_support()
def ts_data2ret(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, shift1: int =-1, shift2: int =-2,
              fill: bool = False) -> pd.DataFrame:
    """
    价格转收益率时序算子 - 计算价格序列的收益率
    return_rate = data_shift2 / data_shift1 - 1

    :param data: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，可以是任意数值列
    :param shift1: int, the value shift as denominator
    :param shift2: int, the value shift as numerator
    :param fill: bool, 是否用0填充NaN值
    :return: pd.DataFrame
    """
    # 输入验证
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("DataFrame input must have a MultiIndex with exactly 2 levels")
    
    if isinstance(data, pd.DataFrame):
        # 使用level=1按股票分组进行时序操作，与其他时序算子保持一致
        shift_1 = data.groupby(level=1).shift(shift1)
        shift_2 = data.groupby(level=1).shift(shift2)
        ret = shift_2 / shift_1 - 1
    else:
        # 对于GroupBy对象，直接应用shift操作
        shift_1 = data.shift(shift1)
        shift_2 = data.shift(shift2)
        ret = shift_2 / shift_1 - 1
    
    # 获取输入数据的列名并为每列生成对应的新列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns.tolist()
    elif hasattr(data, 'obj'):
        # 对于 DataFrameGroupBy，直接使用结果的列名
        original_columns = ret.columns.tolist()
    else:
        original_columns = ['data']
    
    # 修改列名格式 - 为每个原始列生成对应的新列名
    if isinstance(ret, pd.DataFrame):
        ret.columns = [f'ts_data2ret({col}, {shift1}, {shift2})' for col in original_columns]
    
    if fill:
        ret.fillna(0, inplace=True)
    return ret



@multi_period_support()
@df_scalar_broadcast_support()
def ts_delay(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    时序延迟算子 - 返回 n_period 天前的数据
    
    :param data: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，输入数据
    :param n_period: int, 延迟的期数
    :return: pd.DataFrame
    """
    # 输入验证
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("DataFrame input must have a MultiIndex with exactly 2 levels")
    
    def delay(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.shift(n_period)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: delay(x))
        # 修改列名为 ts_delay(原列名, n_period) 的形式
        result.columns = [f'ts_delay({col}, {n_period})' for col in data.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: delay(x))
        # 修改列名为 ts_delay(原列名, n_period) 的形式
        original_columns = res.columns.tolist()
        res.columns = [f'ts_delay({col}, {n_period})' for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_delta(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    时序差分算子 - 计算当前值与 n_period 天前值的差
    Returns data - ts_delay(data, n_period)
    
    :param data: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，输入数据
    :param n_period: int, 差分的期数
    :return: pd.DataFrame
    """
    # 输入验证
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("DataFrame input must have a MultiIndex with exactly 2 levels")
    
    def delta(feature: pd.DataFrame) -> pd.DataFrame:
        return feature - feature.shift(n_period)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: delta(x))
        # 修改列名为 ts_delta(原列名, n_period) 的形式
        result.columns = [f'ts_delta({col}, {n_period})' for col in data.columns]
        return result
    else:
        result: pd.DataFrame = data.transform(lambda x: delta(x))
        # 修改列名为 ts_delta(原列名, n_period) 的形式
        original_columns = result.columns.tolist()
        result.columns = [f'ts_delta({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_returns(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    时序收益率算子 - 计算相对变化率
    Returns the relative change of data over n_period.
    
    :param data: pd.DataFrame或pd.core.groupby.DataFrameGroupBy，输入数据
    :param n_period: int, 计算收益率的期数
    :return: pd.DataFrame
    """
    # 输入验证
    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels != 2:
            raise ValueError("DataFrame input must have a MultiIndex with exactly 2 levels")
    
    # 保存原始列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns.tolist()
    else:
        # 对于 GroupBy 对象，获取原始 DataFrame 的列名
        original_columns = data.obj.columns.tolist()
    
    # 直接计算收益率，不使用 ts_delta 和 ts_delay 以避免列名冲突
    def returns_calc(feature: pd.DataFrame) -> pd.DataFrame:
        delayed = feature.shift(n_period)
        delta = feature - delayed
        return delta / delayed

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: returns_calc(x))
    else:
        result = data.transform(lambda x: returns_calc(x))
    
    # 修改列名为 ts_returns(原列名, n_period) 的形式
    result.columns = [f'ts_returns({col}, {n_period})' for col in original_columns]
    
    return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_sum(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动求和
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动求和结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def sum_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).sum()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: sum_func(x))
        # 修改列名为 ts_sum(原列名, n_period) 的形式
        result.columns = [f'ts_sum({col}, {n_period})' for col in data.columns]
        return result
    else:
        result: pd.DataFrame = data.transform(lambda x: sum_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = result.columns.tolist()
        result.columns = [f'ts_sum({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_product(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动乘积
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动乘积结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
        
    Note: 
        当数据不足以形成完整的滚动窗口时（前 n_period-1 天），返回 NaN，
        确保数据完整性并避免误导性结果。
    """
    def product_func(feature: pd.DataFrame) -> pd.DataFrame:
        # Use rolling with min_periods=n_period to ensure NaN for insufficient data
        return feature.rolling(n_period, min_periods=n_period).apply(lambda x: x.prod(), raw=False)

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: product_func(x))
        # 修改列名为 ts_product(原列名, n_period) 的形式
        result.columns = [f'ts_product({col}, {n_period})' for col in data.columns]
        return result
    else:
        result: pd.DataFrame = data.transform(lambda x: product_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = result.columns.tolist()
        result.columns = [f'ts_product({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_max(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动最大值
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动最大值结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def max_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).max()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: max_func(x))
        # 修改列名为 ts_max(原列名, n_period) 格式
        result.columns = [f"ts_max({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: max_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_max({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_min(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动最小值
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动最小值结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def min_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).min()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: min_func(x))
        # 修改列名为 ts_min(原列名, n_period) 格式
        result.columns = [f"ts_min({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: min_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_min({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_mean(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动平均值
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动平均值结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def mean_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).mean()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: mean_func(x))
        # 修改列名为 ts_mean(原列名, n_period) 格式
        result.columns = [f"ts_mean({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: mean_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_mean({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_std(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动标准差
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动标准差结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def std_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).std()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: std_func(x))
        # 修改列名为 ts_std(原列名, n_period) 格式
        result.columns = [f"ts_std({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: std_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_std({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_dstd(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的下行标准差
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 下行标准差结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """

    def downside_std(df: pd.DataFrame):
        downside_data = df.where(df > 0, np.nan)
        return downside_data.rolling(n_period, min_periods=2).std()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: downside_std(x))
        # 修改列名为 ts_dstd(原列名, n_period) 格式
        result.columns = [f"ts_dstd({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: downside_std(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_dstd({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_kurt(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动峰度
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动峰度结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def kurt_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).kurt()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: kurt_func(x))
        # 修改列名为 ts_kurt(原列名, n_period) 格式
        result.columns = [f"ts_kurt({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: kurt_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_kurt({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_skew(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动偏度
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动偏度结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def skew_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).skew()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: skew_func(x))
        # 修改列名为 ts_skew(原列名, n_period) 格式
        result.columns = [f"ts_skew({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: skew_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_skew({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_median(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动中位数
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动中位数结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def median_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).median()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: median_func(x))
        # 修改列名为 ts_median(原列名, n_period) 格式
        result.columns = [f"ts_median({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: median_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_median({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_rank(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动排名
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动排名结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def rank_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).rank()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: rank_func(x))
        # 修改列名格式
        result.columns = [f"ts_rank({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: rank_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_rank({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_variance(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的滚动方差
    
    Args:
        data: 输入数据，DataFrame 或 DataFrameGroupBy 对象
        n_period: 滚动窗口大小
        
    Returns:
        DataFrame: 滚动方差结果
        
    Raises:
        ValueError: 当输入 DataFrame 不具有 MultiIndex 或索引层级不为 2 时
    """
    def variance_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).var()

    if isinstance(data, pd.DataFrame):
        # 验证输入数据
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("输入 DataFrame 必须具有 MultiIndex")
        if data.index.nlevels != 2:
            raise ValueError("输入 DataFrame 的索引必须有两个层级")
            
        result = data.groupby(level=1).transform(lambda x: variance_func(x))
        # 修改列名格式
        result.columns = [f"ts_variance({col}, {n_period})" for col in result.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: variance_func(x))
        # 使用结果的列名，而不是强制设置索引名称
        original_columns = res.columns.tolist()
        res.columns = [f"ts_variance({col}, {n_period})" for col in original_columns]
        return res



@multi_period_support()
@df_scalar_broadcast_support()
def ts_quantile_up(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的上四分位数（75%分位数）
    
    参数:
    - data: 输入数据，可以是 DataFrame 或 DataFrameGroupBy 对象
    - n_period: 时间窗口大小
    
    返回:
    - DataFrame: 包含上四分位数计算结果的数据框
    """
    def quantile_up_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).quantile(0.75)

    if isinstance(data, pd.DataFrame):
        # 输入验证
        if isinstance(data.index, pd.MultiIndex):
            if data.index.nlevels != 2:
                raise ValueError("DataFrame 必须有两层索引 (datetime, instrument)")
        
        result = data.groupby(level=1).transform(lambda x: quantile_up_func(x))
        # 修改列名格式
        result.columns = [f"ts_quantile_up({col}, {n_period})" for col in data.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: quantile_up_func(x))
        # 修改列名格式，使用原始数据的列名
        res.columns = [f"ts_quantile_up({col}, {n_period})" for col in data.obj.columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_quantile_down(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算过去 n_period 天数据的下四分位数（25%分位数）
    
    参数:
    - data: 输入数据，可以是 DataFrame 或 DataFrameGroupBy 对象
    - n_period: 时间窗口大小
    
    返回:
    - DataFrame: 包含下四分位数计算结果的数据框
    """
    def quantile_down_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).quantile(0.25)

    if isinstance(data, pd.DataFrame):
        # 输入验证
        if isinstance(data.index, pd.MultiIndex):
            if data.index.nlevels != 2:
                raise ValueError("DataFrame 必须有两层索引 (datetime, instrument)")
        
        result = data.groupby(level=1).transform(lambda x: quantile_down_func(x))
        # 修改列名格式
        result.columns = [f"ts_quantile_down({col}, {n_period})" for col in data.columns]
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: quantile_down_func(x))
        # 修改列名格式，使用原始数据的列名
        res.columns = [f"ts_quantile_down({col}, {n_period})" for col in data.obj.columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_zscore(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean:
    (data - ts_mean(data,n_period)) / ts_std(data,n_period)
    """
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def zscore_func(x):
            mean_val = x.rolling(n_period).mean()
            std_val = x.rolling(n_period).std()
            return (x - mean_val) / std_val
        
        result = data.groupby(level=1).transform(zscore_func)
        # 修改列名格式
        result.columns = [f"ts_zscore({col}, {n_period})" for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def zscore_func(x):
            mean_val = x.rolling(n_period).mean()
            std_val = x.rolling(n_period).std()
            return (x - mean_val) / std_val
        
        result = data.transform(zscore_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f"ts_zscore({col}, {n_period})" for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_robust_zscore(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def robust_zscore_func(x):
            med = x.rolling(n_period).median()
            abs_diff = (x - med).abs()
            mad = abs_diff.rolling(n_period).median() * 1.4826
            return (x - med) / mad
        
        result = data.groupby(level=1).transform(robust_zscore_func)
        # 修改列名格式
        result.columns = [f"ts_robust_zscore({col}, {n_period})" for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def robust_zscore_func(x):
            med = x.rolling(n_period).median()
            abs_diff = (x - med).abs()
            mad = abs_diff.rolling(n_period).median() * 1.4826
            return (x - med) / mad
        
        result = data.transform(robust_zscore_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f"ts_robust_zscore({col}, {n_period})" for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_scale(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def scale_func(x):
            min_val = x.rolling(n_period).min()
            max_val = x.rolling(n_period).max()
            return (x - min_val) / (max_val - min_val)
        
        result = data.groupby(level=1).transform(scale_func)
        # 修改列名为 ts_scale(原列名, n_period) 的形式
        result.columns = [f'ts_scale({col}, {n_period})' for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def scale_func(x):
            min_val = x.rolling(n_period).min()
            max_val = x.rolling(n_period).max()
            return (x - min_val) / (max_val - min_val)
        
        result = data.transform(scale_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f'ts_scale({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_sharpe(data: pd.core.groupby.DataFrameGroupBy  | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    计算夏普比率：ts_mean(data, n_period) / ts_std(data, n_period)
    
    参数:
    - data: 输入数据，可以是 DataFrame 或 DataFrameGroupBy 对象
    - n_period: 时间窗口大小
    
    返回:
    - DataFrame: 包含夏普比率计算结果的数据框
    """
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def sharpe_func(x):
            mean_val = x.rolling(n_period).mean()
            std_val = x.rolling(n_period).std()
            return mean_val / std_val
        
        result = data.groupby(level=1).transform(sharpe_func)
        result.columns = [f'ts_sharpe({col}, {n_period})' for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def sharpe_func(x):
            mean_val = x.rolling(n_period).mean()
            std_val = x.rolling(n_period).std()
            return mean_val / std_val
        
        result = data.transform(sharpe_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f'ts_sharpe({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_av_diff(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Returns data - ts_mean(data, n_period)
    """
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def av_diff_func(x):
            mean_val = x.rolling(n_period).mean()
            return x - mean_val
        
        result = data.groupby(level=1).transform(av_diff_func)
        result.columns = [f'ts_av_diff({col}, {n_period})' for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def av_diff_func(x):
            mean_val = x.rolling(n_period).mean()
            return x - mean_val
        
        result = data.transform(av_diff_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f'ts_av_diff({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_max_diff(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Returns data - ts_max(data, n_period)
    """
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def max_diff_func(x):
            max_val = x.rolling(n_period).max()
            return x - max_val
        
        result = data.groupby(level=1).transform(max_diff_func)
        result.columns = [f'ts_max_diff({col}, {n_period})' for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def max_diff_func(x):
            max_val = x.rolling(n_period).max()
            return x - max_val
        
        result = data.transform(max_diff_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f'ts_max_diff({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_min_diff(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Returns data - ts_min(data, n_period)
    """
    if isinstance(data, pd.DataFrame):
        # 对于DataFrame输入，使用groupby transform方法
        def min_diff_func(x):
            min_val = x.rolling(n_period).min()
            return x - min_val
        
        result = data.groupby(level=1).transform(min_diff_func)
        result.columns = [f'ts_min_diff({col}, {n_period})' for col in data.columns]
        return result
    else:
        # 对于DataFrameGroupBy输入，使用transform方法
        def min_diff_func(x):
            min_val = x.rolling(n_period).min()
            return x - min_val
        
        result = data.transform(min_diff_func)
        # 获取实际处理的列名
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        result.columns = [f'ts_min_diff({col}, {n_period})' for col in original_columns]
        return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_corr(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int, rank: bool = False) -> pd.DataFrame:
    """
    计算过去 n_period 天内 x1 和 x2 的相关系数
    
    参数:
    - x1: 第一个输入数据，可以是 DataFrame 或 DataFrameGroupBy 对象
    - x2: 第二个输入数据，可以是 DataFrame 或 DataFrameGroupBy 对象
    - n_period: 时间窗口大小
    - rank: 是否计算 Spearman 相关系数（基于排名），默认为 False（Pearson 相关系数）
    
    返回:
    - DataFrame: 包含相关系数计算结果的数据框
    
    支持多周期计算：当n_period传入列表时，返回包含多个周期结果的DataFrame
    例如：ts_corr(close, volume, [5, 10, 20]) 返回 DataFrame with columns: corr_5, corr_10, corr_20
    """
    # 处理GroupBy对象，转换为DataFrame
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    
    # 保留原始多列 DataFrame 以便后续执行笛卡尔积式计算
    x1_all = x1 if isinstance(x1, pd.DataFrame) else (x1.to_frame(x1.name) if isinstance(x1, pd.Series) else x1)
    x2_all = x2 if isinstance(x2, pd.DataFrame) else (x2.to_frame(x2.name) if isinstance(x2, pd.Series) else x2)
    
    
    # 将DataFrame转换为Series（取第一列）
    if isinstance(x1, pd.DataFrame):
        x1 = x1.iloc[:, 0]
    if isinstance(x2, pd.DataFrame):
        x2 = x2.iloc[:, 0]
    
    # 使用向量化滚动统计公式，替代 groupby.apply + rolling.corr
    def _rolling_corr_series(s1: pd.Series, s2: pd.Series, window: int, use_rank: bool) -> pd.Series:
        if use_rank:
            s1 = s1.groupby(level=1).transform(lambda x: x.rank())
            s2 = s2.groupby(level=1).transform(lambda x: x.rank())

        g1 = s1.groupby(level=1)
        g2 = s2.groupby(level=1)

        mu1 = g1.transform(lambda x: x.rolling(window).mean())
        mu2 = g2.transform(lambda x: x.rolling(window).mean())
        mu12 = (s1 * s2).groupby(level=1).transform(lambda x: x.rolling(window).mean())

        m2_1 = (s1 * s1).groupby(level=1).transform(lambda x: x.rolling(window).mean())
        m2_2 = (s2 * s2).groupby(level=1).transform(lambda x: x.rolling(window).mean())

        var1 = m2_1 - mu1 * mu1
        var2 = m2_2 - mu2 * mu2
        denom = (var1 * var2) ** 0.5
        denom = denom.replace(0, np.nan)
        corr = (mu12 - mu1 * mu2) / denom
        return corr

    x1.name = "feature"
    x2.name = "label"
    res = _rolling_corr_series(x1, x2, n_period, rank)
    result = res.sort_index().to_frame(name='correlation')
    
    # 修改列名格式为 ts_corr(x1_col, x2_col, n_period)
    x1_name = x1_all.columns[0] if isinstance(x1_all, pd.DataFrame) else (x1.name if hasattr(x1, 'name') and x1.name else 'x1')
    x2_name = x2_all.columns[0] if isinstance(x2_all, pd.DataFrame) else (x2.name if hasattr(x2, 'name') and x2.name else 'x2')
    new_columns = [f'ts_corr({x1_name}, {x2_name}, {n_period}, rank:{rank})' for _ in result.columns]
    result.columns = new_columns
    
    # 若存在多列输入，针对所有列对执行笛卡尔积式计算并拼接结果列
    if isinstance(x1_all, pd.DataFrame) and isinstance(x2_all, pd.DataFrame):
        extra_cols = []
        for c1 in x1_all.columns:
            for c2 in x2_all.columns:
                # 跳过已计算的首列对
                if c1 == x1_all.columns[0] and c2 == x2_all.columns[0]:
                    continue
                s1 = x1_all[c1].rename('feature')
                s2 = x2_all[c2].rename('label')
                series = _rolling_corr_series(s1, s2, n_period, rank)
                col_name = f"ts_corr({c1}, {c2}, {n_period}, rank:{rank})"
                extra_cols.append(series.rename(col_name))
        if extra_cols:
            result = pd.concat([result] + extra_cols, axis=1).sort_index()
    
    return result



@multi_period_support() 
@df_scalar_broadcast_support()
def ts_cov(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Returns covariance of data[feature] and data[label] for the past n_period days
    """
    # 处理GroupBy对象
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    
    # 保留原始多列 DataFrame 以便后续执行笛卡尔积式计算
    x1_all = x1 if isinstance(x1, pd.DataFrame) else (x1.to_frame(x1.name) if isinstance(x1, pd.Series) else x1)
    x2_all = x2 if isinstance(x2, pd.DataFrame) else (x2.to_frame(x2.name) if isinstance(x2, pd.Series) else x2)

    # 确保输入是Series（取第一列）
    if isinstance(x1, pd.DataFrame):
        x1 = x1.iloc[:, 0]
    if isinstance(x2, pd.DataFrame):
        x2 = x2.iloc[:, 0]
        
    x1.name = "feature"
    x2.name = "label"
    concat_df = pd.concat([x1, x2], axis=1)
    cov = concat_df.groupby(level=1).apply(lambda x:
                                           x["feature"].rolling(n_period).cov(x["label"])).reset_index(0, drop=True)
    # 确保返回 DataFrame
    result = cov.sort_index()
    if isinstance(result, pd.Series):
        result = result.to_frame(name='covariance')
    
    # 修改列名格式为 ts_cov(x1_col, x2_col, n_period)
    x1_name = x1_all.columns[0] if isinstance(x1_all, pd.DataFrame) else (x1.name if hasattr(x1, 'name') and x1.name else 'x1')
    x2_name = x2_all.columns[0] if isinstance(x2_all, pd.DataFrame) else (x2.name if hasattr(x2, 'name') and x2.name else 'x2')
    new_columns = [f'ts_cov({x1_name}, {x2_name}, {n_period})' for _ in result.columns]
    result.columns = new_columns
    
    # 若存在多列输入，针对所有列对执行笛卡尔积式计算并拼接结果列
    if isinstance(x1_all, pd.DataFrame) and isinstance(x2_all, pd.DataFrame):
        extra_cols = []
        for c1 in x1_all.columns:
            for c2 in x2_all.columns:
                # 跳过已计算的首列对
                if c1 == x1_all.columns[0] and c2 == x2_all.columns[0]:
                    continue
                s1 = x1_all[c1].rename('feature')
                s2 = x2_all[c2].rename('label')
                concat_df2 = pd.concat([s1, s2], axis=1)
                series = concat_df2.groupby(level=1).apply(
                    lambda x: x['feature'].rolling(n_period).cov(x['label'])
                ).reset_index(0, drop=True)
                col_name = f"ts_cov({c1}, {c2}, {n_period})"
                extra_cols.append(series.rename(col_name))
        if extra_cols:
            result = pd.concat([result] + extra_cols, axis=1).sort_index()
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_beta(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Returns beta of data[feature] and data[label] for the past n_period days
    """
    cov = ts_cov(x1, x2, n_period)
    var = ts_variance(x1, n_period)
    
    # 确保列名对齐以进行正确的除法运算
    cov_values = cov.iloc[:, 0]  # 取协方差列的值
    var_values = var.iloc[:, 0]  # 取方差列的值
    
    # 计算 beta
    beta = cov_values / var_values
    
    # 转换为 DataFrame
    result = beta.to_frame(name='beta')
    
    # 修改列名格式为 ts_beta(x1_col, x2_col, n_period)
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1_obj = x1.obj
    else:
        x1_obj = x1
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2_obj = x2.obj
    else:
        x2_obj = x2
    
    if isinstance(x1_obj, pd.DataFrame):
        x1_name = x1_obj.columns[0]
    else:
        x1_name = x1_obj.name if hasattr(x1_obj, 'name') and x1_obj.name else 'x1'
    
    if isinstance(x2_obj, pd.DataFrame):
        x2_name = x2_obj.columns[0]
    else:
        x2_name = x2_obj.name if hasattr(x2_obj, 'name') and x2_obj.name else 'x2'
    
    new_columns = [f'ts_beta({x1_name}, {x2_name}, {n_period})' for _ in result.columns]
    result.columns = new_columns
    
    return result.sort_index()


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_regression(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int, rettype: int = 0) -> pd.DataFrame:
    """
    Returns results of linear model y = beta * x + alpha + resid

    :param x1:
    :param x2:
    :param n_periods:
    :param rettype: 0 for resid, 1 for beta, 2 for alpha, 3 for y_hat, 4 for R^2
    :return:
    """
    # 处理GroupBy对象
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1_data = x1.obj
    else:
        x1_data = x1
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2_data = x2.obj
    else:
        x2_data = x2
    
    # 确保输入是Series（取第一列）
    if isinstance(x1_data, pd.DataFrame):
        x1_series = x1_data.iloc[:, 0]
    else:
        x1_series = x1_data
    if isinstance(x2_data, pd.DataFrame):
        x2_series = x2_data.iloc[:, 0]
    else:
        x2_series = x2_data
        
    x1_series.name = "feature"
    x2_series.name = "label"
    concat_df = pd.concat([x1_series, x2_series], axis=1)
    
    # 计算beta
    cov = concat_df.groupby(level=1).apply(lambda x:
                                           x["feature"].rolling(n_period).cov(x["label"])).reset_index(0, drop=True)
    var = concat_df.groupby(level=1).apply(lambda x:
                                           x["feature"].rolling(n_period).var()).reset_index(0, drop=True)
    beta = cov / var
    
    # 计算alpha
    mean_x1 = concat_df.groupby(level=1).apply(lambda x:
                                               x["feature"].rolling(n_period).mean()).reset_index(0, drop=True)
    mean_x2 = concat_df.groupby(level=1).apply(lambda x:
                                               x["label"].rolling(n_period).mean()).reset_index(0, drop=True)
    alpha = mean_x2 - beta * mean_x1
    
    # 获取原始列名
    if isinstance(x1_data, pd.DataFrame):
        x1_name = x1_data.columns[0]
    else:
        x1_name = x1_data.name if hasattr(x1_data, 'name') and x1_data.name else 'x1'
    
    if isinstance(x2_data, pd.DataFrame):
        x2_name = x2_data.columns[0]
    else:
        x2_name = x2_data.name if hasattr(x2_data, 'name') and x2_data.name else 'x2'
    
    if rettype == 0:
        # resid
        predict = beta * x1_series + alpha
        resid = x2_series - predict
        result = pd.DataFrame({'resid': resid}).sort_index()
        result.columns = [f'ts_regression({x1_name}, {x2_name}, {n_period}, resid)']
        return result
    elif rettype == 1:
        # beta
        result = pd.DataFrame({'beta': beta}).sort_index()
        result.columns = [f'ts_regression({x1_name}, {x2_name}, {n_period}, beta)']
        return result
    elif rettype == 2:
        # alpha
        result = pd.DataFrame({'alpha': alpha}).sort_index()
        result.columns = [f'ts_regression({x1_name}, {x2_name}, {n_period}, alpha)']
        return result
    elif rettype == 3:
        # y_hat
        predict = beta * x1_series + alpha
        result = pd.DataFrame({'y_hat': predict}).sort_index()
        result.columns = [f'ts_regression({x1_name}, {x2_name}, {n_period}, y_hat)']
        return result
    else:
        # R^2
        predict = beta * x1_series + alpha
        predict.name = "predict"
        x2_series.name = "actual"
        predict_df = pd.concat([predict, x2_series], axis=1)
        r_squared = predict_df.groupby(level=1).apply(lambda x:
                                                      x["predict"].rolling(n_period).corr(x["actual"])).reset_index(0, drop=True) ** 2
        result = pd.DataFrame({'r_squared': r_squared}).sort_index()
        result.columns = [f'ts_regression({x1_name}, {x2_name}, {n_period}, r^2)']
        return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_pos_count(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Count positive values of data for the past n_period days.
    """
    # 参数验证：确保n_period为正整数
    if not isinstance(n_period, int) or n_period <= 0:
        raise ValueError(f"n_period must be a positive integer, got {n_period}")
    
    def pos_count_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).apply(lambda x: (x > 0).sum(), raw=False)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: pos_count_func(x))
    else:
        result: pd.DataFrame = data.transform(lambda x: pos_count_func(x))
        # 保持原有的索引名称，不强制设置为["datetime", "instrument"]
        if hasattr(data, 'obj') and data.obj.index.names:
            result.index.names = data.obj.index.names
    
    # 获取输入数据的列名并为每列生成对应的新列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns.tolist()
    elif hasattr(data, 'obj'):
        # 对于GroupBy对象，获取实际选择的列名，而不是原始DataFrame的所有列名
        if hasattr(data, '_obj_with_exclusions'):
            original_columns = data._obj_with_exclusions.columns.tolist()
        else:
            original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']
    
    # 修改列名格式 - 为每个原始列生成对应的新列名
    if isinstance(result, pd.DataFrame):
        result.columns = [f'ts_pos_count({col}, {n_period})' for col in original_columns]
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_neg_count(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Count negative values of data for the past n_period days.
    """
    # 参数验证：确保n_period为正整数
    if not isinstance(n_period, int) or n_period <= 0:
        raise ValueError(f"n_period must be a positive integer, got {n_period}")
    
    def neg_count_func(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).apply(lambda x: (x < 0).sum(), raw=False)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: neg_count_func(x))
    else:
        result: pd.DataFrame = data.transform(lambda x: neg_count_func(x))
        # 保持原有的索引名称，不强制设置为["datetime", "instrument"]
        if hasattr(data, 'obj') and data.obj.index.names:
            result.index.names = data.obj.index.names
    
    # 获取原始列名并格式化
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        # 对于GroupBy对象，获取实际选择的列名，而不是原始DataFrame的所有列名
        if hasattr(data, '_obj_with_exclusions'):
            original_columns = data._obj_with_exclusions.columns
        else:
            original_columns = data.obj.columns
    
    # 为每个原始列生成新的列名
    new_columns = [f"ts_neg_count({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_decay(data:  pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Returns the linear decay on data for the past n_period days.
    """
    def decay(feature: pd.DataFrame) -> pd.DataFrame:
        arr = np.arange(1, n_period + 1)
        weights = arr / sum(arr)
        return feature.rolling(n_period).apply(lambda y: np.dot(y, weights), raw=True)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: decay(x))
    else:
        result: pd.DataFrame = data.transform(lambda x: decay(x))
        # 保持原有的索引名称，不强制设置为["datetime", "instrument"]
        if hasattr(data, 'obj') and data.obj.index.names:
            result.index.names = data.obj.index.names
    
    # 获取原始列名并格式化
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    # 为每个原始列生成新的列名
    new_columns = [f"ts_decay({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support()
@df_scalar_broadcast_support()
def ts_hump(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, hump: float = 0.01) -> pd.DataFrame:
    """
    ts_hump: 限制Alpha的日内变动幅度与频率（降低换手率与回撤）

    核心逻辑：
    - y_t = x_{t-1}（昨日值）
    - Δ_t = x_t - y_t（当日变化）
    - L_t(date, col) = hump × sum_{asset}(|x_t(date, col)|)（按日期的截面绝对值之和，作为全市场阈值）
    - 若 |Δ_t| < L_t，则输出 y_t；否则输出 y_t + sign(Δ_t) × L_t。

    输入支持 DataFrame 或 DataFrameGroupBy；输出始终为 DataFrame。
    列名格式：ts_hump(<col>, <hump>)。
    """
    # 统一 DataFrame 视图
    if isinstance(data, pd.core.groupby.DataFrameGroupBy):
        ref_df = data.obj
        # 时间序列上的昨日值（按资产分组）
        yest = data.transform(lambda x: x.shift(1))
        # 保持原有索引名称
        if hasattr(data, 'obj') and data.obj.index.names:
            yest.index.names = data.obj.index.names
    else:
        ref_df = data
        yest = data.groupby(level=1).transform(lambda x: x.shift(1))

    # 首日填充：无昨日值时，使用当日原值，确保输出不为NaN
    yest_filled = yest.fillna(ref_df)

    # 当日变化
    # 使用原始数值避免算子重载导致的临时列名变更
    original_columns = (ref_df.columns if isinstance(ref_df, pd.DataFrame) else ref_df.obj.columns)
    delta = pd.DataFrame(ref_df.values - yest_filled.values, index=ref_df.index, columns=original_columns)

    # 截面阈值：按日期对绝对值求和，并广播到各资产
    # 这会产生与输入同形的 DataFrame，其中每个日期、每列的所有资产值都相同（该日期的截面总和）
    limit = hump * ref_df.groupby(level=0).transform(lambda x: np.abs(x).sum())

    # 应用门限规则
    delta_abs = pd.DataFrame(np.abs(delta.values), index=delta.index, columns=original_columns)
    mask_small = delta_abs < limit
    # 计算方向并组装输出（使用底层数组避免重载副作用）
    sign_arr = np.sign(delta.values)
    other_vals = yest_filled.values + sign_arr * limit.values
    out = pd.DataFrame(np.where(mask_small.values, yest_filled.values, other_vals),
                       index=yest_filled.index, columns=original_columns)

    # 格式化列名
    out.columns = [f"ts_hump({col}, {hump})" for col in original_columns]
    return out


@multi_period_support()
@df_scalar_broadcast_support()
def ts_hump_decay(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, p: float = 0.01, relative: bool = False) -> pd.DataFrame:
    """
    ts_hump_decay: 忽略小幅变化（相对/绝对两种模式）

    - 绝对模式（relative=False）：若 |x_t - x_{t-1}| > p，则返回 x_t；否则返回 x_{t-1}
    - 相对模式（relative=True）：若 |x_t - x_{t-1}| > p × |x_t + x_{t-1}|，则返回 x_t；否则返回 x_{t-1}

    输入支持 DataFrame 或 DataFrameGroupBy；输出始终为 DataFrame。
    列名格式：
    - relative=False: ts_hump_decay(<col>, <p>)
    - relative=True:  ts_hump_decay(<col>, <p>, relative)
    """
    # 统一 DataFrame 视图与昨日值
    if isinstance(data, pd.core.groupby.DataFrameGroupBy):
        ref_df = data.obj
        yest = data.transform(lambda x: x.shift(1))
        if hasattr(data, 'obj') and data.obj.index.names:
            yest.index.names = data.obj.index.names
    else:
        ref_df = data
        yest = data.groupby(level=1).transform(lambda x: x.shift(1))

    # 首日填充为当日值，避免NaN
    yest_filled = yest.fillna(ref_df)

    # 当日变化（使用底层数组避免重载带来的列名污染）
    original_columns = (ref_df.columns if isinstance(ref_df, pd.DataFrame) else ref_df.obj.columns)
    delta = pd.DataFrame(ref_df.values - yest_filled.values, index=ref_df.index, columns=original_columns)

    # 阈值：绝对或相对
    if relative:
        threshold = pd.DataFrame(p * np.abs(ref_df.values + yest_filled.values),
                                 index=ref_df.index, columns=original_columns)
    else:
        # 标量 p 与 DataFrame 进行比较时会广播
        threshold = p

    # 规则应用：大于阈值则取当日值，否则沿用昨日值
    delta_abs = pd.DataFrame(np.abs(delta.values), index=delta.index, columns=original_columns)
    use_today = delta_abs > threshold
    out_vals = np.where(use_today.values, ref_df.values, yest_filled.values)
    out = pd.DataFrame(out_vals, index=ref_df.index, columns=original_columns)

    # 格式化列名
    if relative:
        out.columns = [f"ts_hump_decay({col}, {p}, relative)" for col in original_columns]
    else:
        out.columns = [f"ts_hump_decay({col}, {p})" for col in original_columns]
    return out


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_ewma(data:  pd.core.groupby.DataFrameGroupBy | pd.DataFrame, half_period: int, n_period: int) -> pd.DataFrame:
    """
    Returns the linear Exponential Weighted Moving Average on data for the past n_period days with half-life period of half_period.
    """
    def ewma(x: pd.DataFrame, lamda: int, n: int) -> pd.DataFrame:
        """
        Exponential Weighted Moving Average
        lamda: half-life period
        """
        arr = np.arange(1, n + 1)
        arr = 0.5 ** (arr/lamda)
        weights = arr / sum(arr)
        return x.rolling(n).apply(lambda y: np.dot(y, weights), raw=True)
        
    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: ewma(x, half_period, n_period))
        # 修改列名格式为 ts_ewma(原列名, half_period, n_period)
        new_columns = [f'ts_ewma({col}, half:{half_period}, {n_period})' for col in result.columns]
        result.columns = new_columns
        return result
    else:
        res: pd.DataFrame = data.transform(lambda x: ewma(x, half_period, n_period))
        # 保持原始索引名称
        res.index.names = data.obj.index.names
        # 修改列名格式为 ts_ewma(原列名, half_period, n_period)
        new_columns = [f'ts_ewma({col}, half:{half_period}, {n_period})' for col in res.columns]
        res.columns = new_columns
        return res


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_argmax(data:  pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    def argmax(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).apply(lambda x: np.argmax(x))

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: argmax(x))
    else:
        result = data.transform(lambda x: argmax(x))
        # 保持原有的索引名称，不强制设置为["datetime", "instrument"]
        if hasattr(data, 'obj') and data.obj.index.names:
            result.index.names = data.obj.index.names
    
    # 获取原始列名并格式化
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    # 为每个原始列生成新的列名
    new_columns = [f"ts_argmax({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_argmin(data:  pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    def argmin(feature: pd.DataFrame) -> pd.DataFrame:
        return feature.rolling(n_period).apply(lambda x: np.argmin(x))

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(lambda x: argmin(x))
    else:
        result = data.transform(lambda x: argmin(x))
        # 保持原有的索引名称，不强制设置为["datetime", "instrument"]
        if hasattr(data, 'obj') and data.obj.index.names:
            result.index.names = data.obj.index.names
    
    # 格式化列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    new_columns = [f"ts_argmin({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_cord(feature: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, label: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    The correlation between feature change ratio and label change ratio
    
    :param feature: pd.DataFrame, feature data
    :param label: pd.DataFrame, label data  
    :param n_period: int, rolling window period
    :return: pd.DataFrame, correlation of change ratios
    """
    # 处理GroupBy对象
    if isinstance(feature, pd.core.groupby.DataFrameGroupBy):
        feature_data = feature.obj
    else:
        feature_data = feature
    if isinstance(label, pd.core.groupby.DataFrameGroupBy):
        label_data = label.obj
    else:
        label_data = label
    
    # 确保输入是Series（取第一列）
    if isinstance(feature_data, pd.DataFrame):
        feature_series = feature_data.iloc[:, 0]
    else:
        feature_series = feature_data
    if isinstance(label_data, pd.DataFrame):
        label_series = label_data.iloc[:, 0]
    else:
        label_series = label_data
    
    # 计算延迟数据
    feature_delay = feature_series.groupby(level=1).transform(lambda x: x.shift(1))
    label_delay = label_series.groupby(level=1).transform(lambda x: x.shift(1))
    
    # 计算变化率
    fd = feature_series / feature_delay
    ld = label_series / label_delay
    
    # 计算相关性
    fd.name = "feature"
    ld.name = "label"
    concat_df = pd.concat([fd, ld], axis=1)
    if n_period > 0:
        corr = concat_df.groupby(level=1).apply(
            lambda x: x["feature"].rolling(n_period).corr(x["label"])).reset_index(0, drop=True)
    else:
        corr = concat_df.groupby(level=1).apply(
            lambda x: x["feature"].corr(x["label"])).reset_index(0, drop=True)
    
    # 获取输入数据的列名
    if isinstance(feature, pd.DataFrame):
        feature_name = feature.columns[0]
    elif hasattr(feature, 'obj'):
        feature_name = feature.obj.columns[0]
    else:
        feature_name = 'feature'
        
    if isinstance(label, pd.DataFrame):
        label_name = label.columns[0]
    elif hasattr(label, 'obj'):
        label_name = label.obj.columns[0]
    else:
        label_name = 'label'
    
    # 确保返回 DataFrame
    if isinstance(corr, pd.Series):
        corr = pd.DataFrame(corr, columns=[f'ts_cord({feature_name}, {label_name}, {n_period})'])
    
    return corr.sort_index()


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_psy(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Psychological Line - percentage of days in past n_period days that price go up
    
    :param data: pd.DataFrame or pd.core.groupby.DataFrameGroupBy, price data
    :param n_period: int, rolling window period
    :return: pd.DataFrame, percentage of up days * 100 with column name 'psy'
    """
    # 类型检查
    if not isinstance(data, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy)):
        raise TypeError(f"Expected DataFrame or DataFrameGroupBy, got {type(data)}")
    
    diff = ts_delta(data, 1)
    result = ts_pos_count(diff, n_period) / n_period * 100
    
    # 格式化列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    new_columns = [f"ts_psy({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_rsv(close_data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, 
           high_data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
           low_data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
           n_period: int) -> pd.DataFrame:
    """
    Raw Stochastic Value - calculates the raw stochastic value
    
    :param close_data: pd.DataFrame, close price data
    :param high_data: pd.DataFrame, high price data  
    :param low_data: pd.DataFrame, low price data
    :param n_period: int, rolling window period
    :return: pd.DataFrame, raw stochastic value
    """
    # 类型检查
    for data, name in [(close_data, 'close_data'), (high_data, 'high_data'), (low_data, 'low_data')]:
        if not isinstance(data, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy)):
            raise TypeError(f"{name} must be a DataFrame or DataFrameGroupBy")
    
    # 处理DataFrame类型
    if isinstance(close_data, pd.DataFrame) and isinstance(high_data, pd.DataFrame) and isinstance(low_data, pd.DataFrame):
        # 获取第一列的值进行计算
        close = close_data.iloc[:, 0]
        high = high_data.iloc[:, 0]
        low = low_data.iloc[:, 0]
        
        # 计算滚动最高价和最低价
        rolling_high = high.rolling(window=n_period, min_periods=1).max()
        rolling_low = low.rolling(window=n_period, min_periods=1).min()
        
        # 计算RSV
        rsv = (close - rolling_low) / (rolling_high - rolling_low) * 100
        
        # 获取原始列名并格式化
        close_name = close_data.columns[0]
        high_name = high_data.columns[0]
        low_name = low_data.columns[0]
        result = rsv.to_frame(f'ts_rsv({close_name}, {high_name}, {low_name}, {n_period})')
        
    # 处理DataFrameGroupBy类型
    elif all(isinstance(data, pd.core.groupby.DataFrameGroupBy) for data in [close_data, high_data, low_data]):
        def calculate_rsv(groups):
            close_group, high_group, low_group = groups
            close = close_group.iloc[:, 0]
            high = high_group.iloc[:, 0]
            low = low_group.iloc[:, 0]
            
            # 计算滚动最高价和最低价
            rolling_high = high.rolling(window=n_period, min_periods=1).max()
            rolling_low = low.rolling(window=n_period, min_periods=1).min()
            
            # 计算RSV
            rsv = (close - rolling_low) / (rolling_high - rolling_low) * 100
            return rsv
        
        # 对每个组应用计算
        result_list = []
        for (close_group, high_group, low_group) in zip(close_data, high_data, low_data):
            rsv = calculate_rsv((close_group[1], high_group[1], low_group[1]))
            result_list.append(rsv)
        
        result_series = pd.concat(result_list)
        
        # 获取原始列名并格式化
        close_name = close_data.obj.columns[0]
        high_name = high_data.obj.columns[0]
        low_name = low_data.obj.columns[0]
        result = result_series.to_frame(f'ts_rsv({close_name}, {high_name}, {low_name}, {n_period})')
    
    # 处理混合类型
    else:
        # 统一转换为DataFrame
        close_df = close_data.obj if hasattr(close_data, 'obj') else close_data
        high_df = high_data.obj if hasattr(high_data, 'obj') else high_data
        low_df = low_data.obj if hasattr(low_data, 'obj') else low_data
        
        close = close_df.iloc[:, 0]
        high = high_df.iloc[:, 0]
        low = low_df.iloc[:, 0]
        
        # 计算滚动最高价和最低价
        rolling_high = high.rolling(window=n_period, min_periods=1).max()
        rolling_low = low.rolling(window=n_period, min_periods=1).min()
        
        # 计算RSV
        rsv = (close - rolling_low) / (rolling_high - rolling_low) * 100
        
        # 获取原始列名并格式化
        close_name = close_df.columns[0]
        high_name = high_df.columns[0]
        low_name = low_df.columns[0]
        result = rsv.to_frame(f'ts_rsv({close_name}, {high_name}, {low_name}, {n_period})')
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_cntp(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Count Positive - percentage of days in past n_period days that price go up
    
    :param data: pd.DataFrame, price data
    :param n_period: int, rolling window period
    :return: pd.DataFrame, percentage of up days (0-1 range)
    """
    diff = ts_delta(data, 1)
    result = ts_pos_count(diff, n_period) / n_period
    
    # 格式化列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    new_columns = [f"ts_cntp({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_cntn(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Count Negative - percentage of days in past n_period days that price go down
    
    :param data: pd.DataFrame, price data
    :param n_period: int, rolling window period
    :return: pd.DataFrame, percentage of down days (0-1 range)
    """
    diff = ts_delta(data, 1)
    result = ts_neg_count(diff, n_period) / n_period
    
    # 格式化列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    new_columns = [f"ts_cntn({col}, {n_period})" for col in original_columns]
    result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_sump(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Sum Positive - ratio of positive price changes to total absolute changes in past n_period days
    
    :param data: pd.DataFrame, price data
    :param n_period: int, rolling window period
    :return: pd.DataFrame, ratio of positive changes (0-1 range)
    """
    # 类型检查
    if not isinstance(data, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy)):
        raise TypeError("data must be a DataFrame or DataFrameGroupBy")
    
    diff = ts_delta(data, 1)
    
    # 创建零值DataFrame，确保列名与diff完全匹配
    zeros = diff * 0
    
    result = ts_sum(bigger(diff, zeros), n_period) / ts_sum(abs(diff), n_period)
    
    # 格式化列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    new_columns = [f"ts_sump({col}, {n_period})" for col in original_columns]
    
    # 确保返回DataFrame并设置列名
    if isinstance(result, pd.Series):
        result = result.to_frame(new_columns[0])
    elif isinstance(result, pd.DataFrame):
        result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_sumn(data: pd.core.groupby.DataFrameGroupBy |pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    Sum Negative - ratio of negative price changes to total absolute changes in past n_period days
    
    :param data: pd.DataFrame, price data
    :param n_period: int, rolling window period
    :return: pd.DataFrame, ratio of negative changes (0-1 range)
    """
    # 类型检查
    if not isinstance(data, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy)):
        raise TypeError("data must be a DataFrame or DataFrameGroupBy")
    
    diff = ts_delta(data, 1)
    
    # 创建零值DataFrame，确保列名与diff完全匹配
    zeros = diff * 0
    
    result = ts_sum(bigger(-diff, zeros), n_period) / ts_sum(abs(diff), n_period)
    
    # 格式化列名
    if isinstance(data, pd.DataFrame):
        original_columns = data.columns
    else:
        original_columns = data.obj.columns
    
    new_columns = [f"ts_sumn({col}, {n_period})" for col in original_columns]
    
    # 确保返回DataFrame并设置列名
    if isinstance(result, pd.Series):
        result = result.to_frame(new_columns[0])
    elif isinstance(result, pd.DataFrame):
        result.columns = new_columns
    
    return result


@multi_period_support() 
@df_scalar_broadcast_support()
def ts_wvma(price_data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, 
            volume_data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, 
            n_period: int) -> pd.DataFrame:
    """
    Weighted Volume Moving Average - calculates coefficient of variation of weighted volume
    
    :param price_data: pd.DataFrame, price data
    :param volume_data: pd.DataFrame, volume data  
    :param n_period: int, rolling window period
    :return: pd.DataFrame, coefficient of variation (std/mean) of weighted volume
    """
    # 类型检查
    if not isinstance(price_data, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy)):
        raise TypeError("price_data must be a DataFrame or DataFrameGroupBy")
    if not isinstance(volume_data, (pd.DataFrame, pd.core.groupby.DataFrameGroupBy)):
        raise TypeError("volume_data must be a DataFrame or DataFrameGroupBy")
    
    # 计算权重（价格变化的绝对值）
    weight = abs(ts_returns(price_data, 1))
    
    # 计算加权成交量
    # 多列输入时按列位置一一对应进行计算（不做笛卡尔积）
    if isinstance(weight, pd.DataFrame) and isinstance(volume_data, pd.DataFrame):
        if weight.shape[1] > 1 or volume_data.shape[1] > 1:
            pair_len = min(weight.shape[1], volume_data.shape[1])
            weighted_vol = pd.DataFrame(index=weight.index)
            for i in range(pair_len):
                weighted_vol[i] = weight.iloc[:, i] * volume_data.iloc[:, i]
        else:
            # 单列情况保持原有行为
            weighted_vol = pd.DataFrame(
                weight.iloc[:, 0] * volume_data.iloc[:, 0],
                index=weight.index,
                columns=['weighted_volume']
            )
    elif isinstance(weight, pd.core.groupby.DataFrameGroupBy) and isinstance(volume_data, pd.core.groupby.DataFrameGroupBy):
        # 对于GroupBy对象，使用其底层DataFrame按列位置配对计算
        weight_df = weight.obj
        volume_df = volume_data.obj
        if weight_df.shape[1] > 1 or volume_df.shape[1] > 1:
            pair_len = min(weight_df.shape[1], volume_df.shape[1])
            weighted_vol = pd.DataFrame(index=weight_df.index)
            for i in range(pair_len):
                weighted_vol[i] = weight_df.iloc[:, i] * volume_df.iloc[:, i]
        else:
            weighted_vol = pd.DataFrame(
                weight_df.iloc[:, 0] * volume_df.iloc[:, 0],
                index=weight_df.index,
                columns=['weighted_volume']
            )
    else:
        # 混合类型处理：尽力转换成DataFrame并按单列或配对逻辑处理
        weight_df = weight.obj if hasattr(weight, 'obj') else weight
        volume_df = volume_data.obj if hasattr(volume_data, 'obj') else volume_data
        if isinstance(weight_df, pd.DataFrame) and isinstance(volume_df, pd.DataFrame):
            if weight_df.shape[1] > 1 or volume_df.shape[1] > 1:
                pair_len = min(weight_df.shape[1], volume_df.shape[1])
                weighted_vol = pd.DataFrame(index=weight_df.index)
                for i in range(pair_len):
                    weighted_vol[i] = weight_df.iloc[:, i] * volume_df.iloc[:, i]
            else:
                weighted_vol = pd.DataFrame(
                    weight_df.iloc[:, 0] * volume_df.iloc[:, 0],
                    index=weight_df.index,
                    columns=['weighted_volume']
                )
        else:
            # 无法获取DataFrame，采取保守的单列计算
            raise TypeError("Unsupported input types for ts_wvma: expected DataFrame or DataFrameGroupBy")
    
    # 计算标准差和均值
    vol_std = ts_std(weighted_vol, n_period)
    vol_mean = ts_mean(weighted_vol, n_period)
    # 关键修复：ts_std/ts_mean 会重命名列（ts_std(...), ts_mean(...))
    # 为了按列位置进行除法，需要强制两者拥有一致的列名，以避免按列名对齐导致 NaN 和额外列。
    if isinstance(vol_std, pd.DataFrame) and isinstance(vol_mean, pd.DataFrame):
        # 使用加权成交量的列名作为共同列名
        common_cols = list(weighted_vol.columns)
        # 如果某些函数返回列数和weighted_vol不一致，回退为按序号命名
        if len(common_cols) != vol_std.shape[1] or len(common_cols) != vol_mean.shape[1]:
            common_cols = [i for i in range(max(vol_std.shape[1], vol_mean.shape[1]))][:min(vol_std.shape[1], vol_mean.shape[1])]
            # 对齐形状（通常两者形状一致，这里是保守处理）
            vol_std = vol_std.iloc[:, :len(common_cols)].copy()
            vol_mean = vol_mean.iloc[:, :len(common_cols)].copy()
        vol_std = vol_std.copy(); vol_std.columns = common_cols
        vol_mean = vol_mean.copy(); vol_mean.columns = common_cols
    
    # 避免除零错误，当均值为0时返回NaN；按列对应进行除法
    if isinstance(vol_mean, pd.DataFrame):
        mean_safe = vol_mean.replace(0, np.nan)
    else:
        mean_safe = vol_mean
        mean_safe.iloc[:, 0] = mean_safe.iloc[:, 0].replace(0, np.nan)

    # 保持DataFrame形态进行按列除法（此时列名已经对齐）
    result = vol_std / mean_safe
    
    # 获取输入数据的列名
    if isinstance(price_data, pd.DataFrame):
        price_name = price_data.columns[0]
    elif hasattr(price_data, 'obj'):
        price_name = price_data.obj.columns[0]
    else:
        price_name = 'price'
        
    if isinstance(volume_data, pd.DataFrame):
        volume_name = volume_data.columns[0]
    elif hasattr(volume_data, 'obj'):
        volume_name = volume_data.obj.columns[0]
    else:
        volume_name = 'volume'
    
    # 确保返回DataFrame并设置列名
    if isinstance(result, pd.Series):
        result = result.to_frame(f'ts_wvma({price_name}, {volume_name}, {n_period})')
    elif isinstance(result, pd.DataFrame):
        # 多列时按位置一一对应命名
        if result.shape[1] == 1:
            result.columns = [f'ts_wvma({price_name}, {volume_name}, {n_period})']
        else:
            # 组装每列的名称：使用输入的列名按位置配对
            if isinstance(price_data, pd.DataFrame):
                price_cols = list(price_data.columns)
            elif hasattr(price_data, 'obj'):
                price_cols = list(price_data.obj.columns)
            else:
                price_cols = [price_name] * result.shape[1]

            if isinstance(volume_data, pd.DataFrame):
                volume_cols = list(volume_data.columns)
            elif hasattr(volume_data, 'obj'):
                volume_cols = list(volume_data.obj.columns)
            else:
                volume_cols = [volume_name] * result.shape[1]

            pair_len = min(len(price_cols), len(volume_cols), result.shape[1])
            new_cols = [f'ts_wvma({price_cols[i]}, {volume_cols[i]}, {n_period})' for i in range(pair_len)]
            # 若某些列无法命名（不太可能），用索引补齐
            if len(new_cols) < result.shape[1]:
                new_cols += [f'ts_wvma({price_name}, {volume_name}, {n_period})_{i}' for i in range(len(new_cols), result.shape[1])]
            result.columns = new_cols
    
    return result


@df_scalar_broadcast_support()
def cs_rank(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    """
    Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0.
    """
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: x.rank(pct=True))
    else:
        res = data.transform(lambda x: x.rank(pct=True))
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_rank({name})")
    else:
        original_columns = list(res.columns)
        new_columns = [f"cs_rank({col})" for col in original_columns]
        res.columns = new_columns
    return res


@df_scalar_broadcast_support()
def cs_zscore(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean
    """
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
    else:
        res = data.transform(lambda x: (x - x.mean()) / x.std())
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_zscore({name})")
    else:
        original_columns = list(res.columns)
        new_columns = [f"cs_zscore({col})" for col in original_columns]
        res.columns = new_columns
    return res


@df_scalar_broadcast_support()
def cs_robust_zscore(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        def robust_zscore_func(x):
            median_val = x.median()
            deviation = x - median_val
            mad = abs(deviation).median() / 1.4826
            return deviation / mad
        res = data.groupby(level=0).transform(robust_zscore_func)
    else:
        def robust_zscore_func(x):
            median_val = x.median()
            deviation = x - median_val
            mad = abs(deviation).median() / 1.4826
            return deviation / mad
        res = data.transform(robust_zscore_func)
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_robust_zscore({name})")
    else:
        original_columns = list(res.columns)
        new_columns = [f"cs_robust_zscore({col})" for col in original_columns]
        res.columns = new_columns
    return res


@df_scalar_broadcast_support()
def cs_scale(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        def scale_func(x):
            min_val = x.min()
            max_val = x.max()
            return (x - min_val) / (max_val - min_val)
        res = data.groupby(level=0).transform(scale_func)
    else:
        def scale_func(x):
            min_val = x.min()
            max_val = x.max()
            return (x - min_val) / (max_val - min_val)
        res = data.transform(scale_func)
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_scale({name})")
    else:
        original_columns = list(res.columns)
        new_columns = [f"cs_scale({col})" for col in original_columns]
        res.columns = new_columns
    return res


@df_scalar_broadcast_support()
def cs_mean(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    """
    This function is not for regular alphas which have two index levels. It calculates the mean value of all instruments
    on a particular time tick. You may use this for calculating the relationship between single instrument and the index
    """
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: x.mean())
    else:
        res = data.transform(lambda x: x.mean())
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_mean({name})")
    else:
        original_columns = list(res.columns)
        res.columns = [f"cs_mean({col})" for col in original_columns]
    return res


@df_scalar_broadcast_support()
def cs_sum(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    """
    Cross-sectional sum per date, broadcast to instrument rows.
    - Accepts DataFrame or DataFrameGroupBy; returns DataFrame
    - Column naming: cs_sum(<col>)
    """
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: x.sum())
    else:
        res = data.transform(lambda x: x.sum())
    # Ensure DataFrame return and set column names
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_sum({name})")
    else:
        original_columns = list(res.columns)
        res.columns = [f"cs_sum({col})" for col in original_columns]
    return res

@df_scalar_broadcast_support()
def cs_std(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: x.std())
    else:
        res = data.transform(lambda x: x.std())
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_std({name})")
    else:
        original_columns = list(res.columns)
        res.columns = [f"cs_std({col})" for col in original_columns]
    return res


@df_scalar_broadcast_support()
def cs_variance(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(lambda x: x.var())
    else:
        res = data.transform(lambda x: x.var())
    # 确保返回 DataFrame，并设置列名
    if isinstance(res, pd.Series):
        name = res.name if hasattr(res, 'name') and res.name else 'data'
        res = res.to_frame(f"cs_variance({name})")
    else:
        original_columns = list(res.columns)
        res.columns = [f"cs_variance({col})" for col in original_columns]
    return res


@df_scalar_broadcast_support()
def cs_cov(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
           x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional covariance per date, broadcast to instrument rows.
    - Accepts DataFrame/Series or GroupBy; converts to DataFrame internally
    - Supports multiple columns by pairing common column names; defaults to first columns otherwise
    - Returns a DataFrame with columns named like: cs_cov(col1, col2)
    """
    # Normalize inputs to DataFrame
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    if isinstance(x1, pd.Series) or isinstance(x2, pd.Series):
        raise TypeError("cs_cov only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    # Determine column pairs (use common names if available)
    cols1 = list(x1.columns)
    cols2 = list(x2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        df_pair = pd.concat([
            x1[[c1]].rename(columns={c1: "feature"}),
            x2[[c2]].rename(columns={c2: "label"})
        ], axis=1)
        # Scalar per date, then broadcast to MultiIndex by aligning level 0
        by_date = df_pair.groupby(level=0, group_keys=False).apply(lambda g: g["feature"].cov(g["label"]))
        broadcast = pd.Series(by_date.reindex(x1.index.get_level_values(0)).to_numpy(), index=x1.index)
        colname = f"cs_cov({c1}, {c2})"
        result_frames.append(broadcast.to_frame(name=colname))

    result = pd.concat(result_frames, axis=1).sort_index()
    return result


@df_scalar_broadcast_support()
def cs_corr(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
            x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
            rank: bool = False) -> pd.DataFrame:
    """
    Cross-sectional correlation per date, broadcast to instrument rows.
    - When rank=True, use Spearman; otherwise Pearson
    - Returns DataFrame with columns named: cs_corr(col1, col2)
    """
    # Normalize inputs to DataFrame
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    if isinstance(x1, pd.Series) or isinstance(x2, pd.Series):
        raise TypeError("cs_corr only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(x1.columns)
    cols2 = list(x2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    method = "spearman" if rank else "pearson"
    result_frames = []
    for c1, c2 in pairs:
        df_pair = pd.concat([
            x1[[c1]].rename(columns={c1: "feature"}),
            x2[[c2]].rename(columns={c2: "label"})
        ], axis=1)
        by_date = df_pair.groupby(level=0, group_keys=False).apply(lambda g: g["feature"].corr(g["label"], method=method))
        broadcast = pd.Series(by_date.reindex(x1.index.get_level_values(0)).to_numpy(), index=x1.index)
        colname = f"cs_corr({c1}, {c2})"
        result_frames.append(broadcast.to_frame(name=colname))

    result = pd.concat(result_frames, axis=1).sort_index()
    return result


@df_scalar_broadcast_support()
def cs_beta(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
            x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional beta per date: cov(x1,x2)/var(x1), broadcast to instrument rows.
    - Pair columns by common names; default to first columns
    - Returns DataFrame with columns: cs_beta(col1, col2)
    """
    # Normalize inputs
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    if isinstance(x1, pd.Series) or isinstance(x2, pd.Series):
        raise TypeError("cs_beta only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(x1.columns)
    cols2 = list(x2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    result_frames = []
    for c1, c2 in pairs:
        # cov per date (scalar) and broadcast
        df_pair = pd.concat([
            x1[[c1]].rename(columns={c1: "feature"}),
            x2[[c2]].rename(columns={c2: "label"})
        ], axis=1)
        by_date = df_pair.groupby(level=0, group_keys=False).apply(lambda g: g["feature"].cov(g["label"]))
        cov_series = pd.Series(by_date.reindex(x1.index.get_level_values(0)).to_numpy(), index=x1.index)
        # variance of x1 per date (broadcast)
        var_series = x1[[c1]].groupby(level=0).transform(lambda s: s.var()).iloc[:, 0]
        beta_series = cov_series / var_series
        colname = f"cs_beta({c1}, {c2})"
        result_frames.append(beta_series.to_frame(name=colname))

    result = pd.concat(result_frames, axis=1).sort_index()
    return result


@df_scalar_broadcast_support()
def cs_alpha(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
             x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional alpha per date from simple regression: alpha = mean(x2) - beta*mean(x1)
    - Pair columns by common names; default to first columns
    - Returns DataFrame with columns: cs_alpha(col1, col2)
    """
    # Normalize inputs
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    if isinstance(x1, pd.Series) or isinstance(x2, pd.Series):
        raise TypeError("cs_alpha only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(x1.columns)
    cols2 = list(x2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    out_cols = {}
    for c1, c2 in pairs:
        beta_series = cs_beta(x1[[c1]], x2[[c2]]).iloc[:, 0]
        mean_x1 = x1[c1].groupby(level=0).transform(lambda s: s.mean())
        mean_x2 = x2[c2].groupby(level=0).transform(lambda s: s.mean())
        alpha_series = mean_x2 - beta_series * mean_x1
        out_cols[f"cs_alpha({c1}, {c2})"] = alpha_series

    result = pd.concat(out_cols.values(), axis=1)
    result.columns = list(out_cols.keys())
    return result.sort_index()


@df_scalar_broadcast_support()
def cs_resid(x1: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
             x2: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional residuals per date: resid = x2 - (beta*x1 + alpha)
    - Pair columns by common names; default to first columns
    - Returns DataFrame with columns: cs_resid(col1, col2)
    """
    # Normalize inputs
    if isinstance(x1, pd.core.groupby.DataFrameGroupBy):
        x1 = x1.obj
    if isinstance(x2, pd.core.groupby.DataFrameGroupBy):
        x2 = x2.obj
    if isinstance(x1, pd.Series) or isinstance(x2, pd.Series):
        raise TypeError("cs_resid only accepts DataFrame or DataFrameGroupBy inputs; Series is not allowed")

    cols1 = list(x1.columns)
    cols2 = list(x2.columns)
    common = [c for c in cols1 if c in cols2]
    pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]

    out_cols = {}
    for c1, c2 in pairs:
        beta_series = cs_beta(x1[[c1]], x2[[c2]]).iloc[:, 0]
        mean_x1 = x1[c1].groupby(level=0).transform(lambda s: s.mean())
        mean_x2 = x2[c2].groupby(level=0).transform(lambda s: s.mean())
        alpha_series = mean_x2 - beta_series * mean_x1
        resid_series = x2[c2] - (beta_series * x1[c1] + alpha_series)
        out_cols[f"cs_resid({c1}, {c2})"] = resid_series

    result = pd.concat(out_cols.values(), axis=1)
    result.columns = list(out_cols.keys())
    return result.sort_index()


@df_scalar_broadcast_support()
def cs_shrink(data: pd.core.groupby.DataFrameGroupBy |  pd.DataFrame) ->  pd.DataFrame:
    """
    Cross-sectional shrink on each date, returning DataFrame and consistent column names.
    Disallows Series inputs; preserves original index names; avoids column length mismatches.
    """
    # 类型检查：不允许 Series
    if isinstance(data, pd.Series):
        raise TypeError("cs_shrink only accepts DataFrame or DataFrameGroupBy; Series is not allowed")

    if isinstance(data, pd.DataFrame):
        # 对于 DataFrame 输入，按日期分组执行两段收缩逻辑
        res = data.groupby(level=0).transform(lambda x: x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5))
        res = res.groupby(level=0).transform(lambda x: x.where(x >= -3, -3 + (x + 3).div(x.min() + 3) * 0.5))
        # 统一列名
        res.columns = [f"cs_shrink({col})" for col in data.columns]
        return res
    else:
        # 对于 GroupBy 输入，使用 transform 在每个分组中执行
        res = data.transform(lambda x: x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5))
        # 第二次变换需按日期分组以保持与 DataFrame 分支一致
        res = res.groupby(level=0).transform(lambda x: x.where(x >= -3, -3 + (x + 3).div(x.min() + 3) * 0.5))
        # 获取原始列名以命名结果列
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"cs_shrink({col})" for col in original_columns]
        return res


@df_scalar_broadcast_support()
def cs_mad_winsor(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional MAD winsorization per date, returning DataFrame with consistent naming.
    Disallows Series inputs; no forced index name changes.
    """
    # 类型检查：不允许 Series
    if isinstance(data, pd.Series):
        raise TypeError("cs_mad_winsor only accepts DataFrame or DataFrameGroupBy; Series is not allowed")

    if isinstance(data, pd.DataFrame):
        # 计算每个日期的中位数与MAD（Median Absolute Deviation）并裁剪
        med = data.groupby(level=0).transform(lambda x: x.median())
        mad = (data.sub(med).abs()).groupby(level=0).transform(lambda x: x.median())
        up = med + 3 * mad * 1.4826
        down = med - 3 * mad * 1.4826
        res = data.clip(upper=up, lower=down)
        res.columns = [f"cs_mad_winsor({col})" for col in data.columns]
        return res
    else:
        # 在分组内执行 winsor 裁剪，并保持返回为 DataFrame
        def winsor_func(x: pd.DataFrame) -> pd.DataFrame:
            med = x.median()
            mad = x.sub(med).abs().median()
            up = med + 3 * mad * 1.4826
            down = med - 3 * mad * 1.4826
            return x.clip(upper=up, lower=down)

        res = data.transform(winsor_func)
        # 命名列
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"cs_mad_winsor({col})" for col in original_columns]
        return res


@df_scalar_broadcast_support()
def cs_winsor(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, std: float = 3.0) -> pd.DataFrame:
    """
    Cross-sectional winsorization per date using mean ± std*k thresholds.
    - Differs from cs_mad_winsor: this uses mean and standard deviation instead of median/MAD.
    - Parameters:
      - std: clipping multiplier k for standard deviation (default 4.0)
    - Returns: DataFrame with column names: cs_winsor(<col>, <std>)
    """
    if isinstance(data, pd.DataFrame):
        # 计算每个日期的均值与标准差（与输入形状对齐）
        mean_df = data.groupby(level=0).transform(lambda x: x.mean())
        std_df = data.groupby(level=0).transform(lambda x: x.std())

        # 使用 numpy 计算阈值，避免被自定义的 DataFrame 算子重载影响列名/配对逻辑
        mean_arr = mean_df.to_numpy()
        std_arr = std_df.to_numpy()
        cond_arr = np.isfinite(std_arr) & (std_arr > 0)

        up_arr = mean_arr + std * std_arr
        down_arr = mean_arr - std * std_arr
        # 标准差为 0 或 NaN 的位置不裁剪（置为 ±inf）
        up_arr = np.where(cond_arr, up_arr, np.inf)
        down_arr = np.where(cond_arr, down_arr, -np.inf)

        # 回到与原数据严格对齐的 DataFrame，以确保 clip 对齐正常
        up_df = pd.DataFrame(up_arr, index=data.index, columns=list(data.columns))
        down_df = pd.DataFrame(down_arr, index=data.index, columns=list(data.columns))

        # 使用 numpy 避开 DataFrame.clip 内部对比较运算符的依赖（已被重载）
        data_arr = data.to_numpy(dtype=float)
        up_arr = up_df.to_numpy(dtype=float)
        down_arr = down_df.to_numpy(dtype=float)
        res_arr = np.minimum(np.maximum(data_arr, down_arr), up_arr)
        res = pd.DataFrame(res_arr, index=data.index, columns=list(data.columns))
        res.columns = [f"cs_winsor({col}, {std})" for col in data.columns]
        return res
    else:
        def winsor_func(x: pd.DataFrame) -> pd.DataFrame:
            # 在分组内：上/下界使用按列的 Series，并由 clip 自动在行维度广播
            m = x.mean()
            s = x.std()
            upper = (m + std * s).where(s.gt(0), np.inf)
            lower = (m - std * s).where(s.gt(0), -np.inf)
            # numpy 逐元素裁剪，Series(m,s) 在列维度广播
            xa = x.to_numpy(dtype=float)
            ua = np.broadcast_to(upper.to_numpy(dtype=float), xa.shape)
            la = np.broadcast_to(lower.to_numpy(dtype=float), xa.shape)
            ra = np.minimum(np.maximum(xa, la), ua)
            return pd.DataFrame(ra, index=x.index, columns=list(x.columns))

        res = data.transform(winsor_func)
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"cs_winsor({col}, {std})" for col in original_columns]
        return res


@df_scalar_broadcast_support()
def cs_quantile(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
                driver: str = 'uniform',
                sigma: float = 1.0) -> pd.DataFrame:
    """
    Map cross-sectional percentile ranks to a target distribution per date.
    - driver: 'uniform' | 'gaussian' | 'cauchy'
      * uniform: return percentile rank in [0,1]
      * gaussian: map to N(0, sigma^2) via inverse CDF (scipy if available, logistic approximation otherwise)
      * cauchy: map to Cauchy(scale=sigma) via tan(pi*(p-0.5))
    - sigma: scale parameter for gaussian/cauchy drivers
    Returns DataFrame with column names: cs_quantile(<col>, <driver>, <sigma>)
    """

    eps = 1e-6

    def map_percentile_to_dist(p: pd.Series) -> pd.Series:
        # clip to avoid infinities
        p = p.clip(eps, 1 - eps)
        if driver.lower() == 'uniform':
            return p
        elif driver.lower() == 'cauchy':
            return np.tan(np.pi * (p - 0.5)) * sigma
        else:
            # gaussian (normal) mapping
            try:
                from scipy.stats import norm
                return pd.Series(norm.ppf(p), index=p.index) * sigma
            except Exception:
                # logistic approximation to normal quantile, scaled to unit variance
                q = np.log(p / (1 - p))
                q = q * (np.sqrt(3) / np.pi)  # scale so Var≈1
                return pd.Series(q, index=p.index) * sigma

    if isinstance(data, pd.DataFrame):
        # percentile rank per date
        pct = data.groupby(level=0).transform(lambda x: x.rank(pct=True))
        res = pct.groupby(level=0).transform(map_percentile_to_dist)
        res.columns = [f"cs_quantile({col}, {driver}, {sigma})" for col in data.columns]
        return res
    else:
        def quantile_func(x: pd.DataFrame) -> pd.DataFrame:
            p = x.rank(pct=True)
            return p.apply(map_percentile_to_dist, axis=0)

        res = data.transform(quantile_func)
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"cs_quantile({col}, {driver}, {sigma})" for col in original_columns]
        return res


@df_scalar_broadcast_support()
def cs_normalize(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
                 useStd: bool = False,
                 limit: float = 0.0) -> pd.DataFrame:
    """
    Cross-sectional normalization per date.
    - Subtract cross-sectional mean on each date.
    - If useStd=True, divide by cross-sectional standard deviation.
    - If limit>0, clip result to [-limit, limit].
    Returns DataFrame with column names: cs_normalize(<col>, <useStd>, <limit>)
    """
    def normalize_block(x: pd.Series | pd.DataFrame):
        # 对 transform 来说，x 会是 Series（按列逐个调用）；
        # 对 group.apply 场景，x 可能是 DataFrame。统一按列处理。
        if isinstance(x, pd.DataFrame):
            return x.apply(normalize_block, axis=0)
        # x 是 Series：计算横截面均值与可选标准化
        m = x.mean()
        y = x - m
        if useStd:
            s = x.std()
            if not np.isfinite(s) or s == 0:
                # 标准差无效时返回全 NaN，避免除零/Inf
                y = pd.Series(np.nan, index=x.index)
            else:
                y = y / s
        if limit and limit > 0:
            # 在标准化结果上进行限制（±limit）
            y = y.clip(lower=-limit, upper=limit)
        return y

    if isinstance(data, pd.DataFrame):
        res = data.groupby(level=0).transform(normalize_block)
        res.columns = [f"cs_normalize({col}, {useStd}, {limit})" for col in data.columns]
        return res
    else:
        res = data.transform(normalize_block)
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"cs_normalize({col}, {useStd}, {limit})" for col in original_columns]
        return res
@df_scalar_broadcast_support()
def ts_days_from_last_change(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame) -> pd.DataFrame:
    """
    逐资产计算“自上次变化以来的天数”。
    - 若当日值与昨日值不同（或任一为 NaN），记为发生变化，当天输出 0；
    - 若当日值与昨日值相同，则在上一日基础上 +1；

    输入支持 DataFrame 和 DataFrameGroupBy；输出为 DataFrame，列名格式：ts_days_from_last_change(<col>)。
    索引假定为 MultiIndex [datetime, instrument]，按 instrument 维度做时间序列运算。
    """
    # 针对 Series 的实现，兼容 groupby.transform 的逐列调用
    def days_since_change_series(s: pd.Series) -> pd.Series:
        prev = s.shift(1)
        changed = (s != prev) | s.isna() | prev.isna()
        group_id = changed.cumsum()
        return s.groupby(group_id).cumcount()

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(days_since_change_series)
        result.columns = [f"ts_days_from_last_change({col})" for col in data.columns]
        return result
    else:
        res = data.transform(days_since_change_series)
        # 命名列使用真实处理的列名（兼容 GroupBy 的 _selection）
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"ts_days_from_last_change({col})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_last_diff_value(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame, n_period: int) -> pd.DataFrame:
    """
    在过去 n_period 天内，返回“最近一次与当前值不同”的历史值（不包含当日）。
    - 若在窗口内未发现与当前值不同的历史值，则返回 NaN。

    列名格式：ts_last_diff_value(<col>, <n_period>)。
    """
    # 针对 Series 的实现，兼容 groupby.transform 的逐列调用
    def last_diff_value_series(s: pd.Series) -> pd.Series:
        def find_last_diff(window: pd.Series):
            if len(window) == 0:
                return np.nan
            curr = window.iloc[-1]
            # 从倒数第二个开始向前寻找与当前值不同的值
            for i in range(len(window) - 2, -1, -1):
                v = window.iloc[i]
                if pd.isna(curr) or pd.isna(v):
                    # 跳过 NaN 值
                    continue
                if v != curr:
                    return v
            return np.nan
        return s.rolling(n_period).apply(find_last_diff, raw=False)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(last_diff_value_series)
        result.columns = [f"ts_last_diff_value({col}, {n_period})" for col in data.columns]
        return result
    else:
        res = data.transform(last_diff_value_series)
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"ts_last_diff_value({col}, {n_period})" for col in original_columns]
        return res


@multi_period_support()
@df_scalar_broadcast_support()
def ts_kth_element(data: pd.core.groupby.DataFrameGroupBy | pd.DataFrame,
                   n_period: int,
                   k: int | str = 1,
                   ignore: str | list | tuple | int | float | None = None) -> pd.DataFrame:
    """
    返回回看窗口（含当日）内从近到远第 k 个“非忽略值”。
    - ignore 支持以空格分隔的标量列表字符串，例如 "NAN 0"；也可传列表/元组/单值。
    - 当 k=1 且 ignore 包含 NAN 时，等价于“后向填充非 NaN 值”的 Backfill。
    - 若在窗口内未找到第 k 个有效值，则返回 NaN。

    列名格式：ts_kth_element(<col>, <n_period>, <k>)。
    """
    # 解析 k
    try:
        k_int = int(k)
    except Exception:
        raise ValueError(f"k 必须为整数或可转换为整数的字符串，当前: {k}")

    # 解析 ignore 参数
    def parse_ignore(ignore_param):
        ignore_nan = False
        ignore_vals: set[float] = set()
        if ignore_param is None:
            return ignore_nan, ignore_vals
        if isinstance(ignore_param, (list, tuple)):
            tokens = list(ignore_param)
        elif isinstance(ignore_param, (int, float)):
            tokens = [ignore_param]
        else:
            tokens = str(ignore_param).strip().split()
        for tok in tokens:
            if isinstance(tok, (int, float)):
                ignore_vals.add(float(tok))
            else:
                t = str(tok).strip().upper()
                if t in {"NAN", "NA", "NONE"}:
                    ignore_nan = True
                else:
                    try:
                        ignore_vals.add(float(tok))
                    except Exception:
                        # 非数字字符串（除 NAN/NA/NONE）忽略
                        pass
        return ignore_nan, ignore_vals

    ignore_nan, ignore_vals = parse_ignore(ignore)

    # 针对 Series 的实现，兼容 groupby.transform 的逐列调用
    def kth_series(s: pd.Series) -> pd.Series:
        def kth_in_window(window: pd.Series):
            cnt = 0
            # 从近到远遍历（包含当日）
            for i in range(len(window) - 1, -1, -1):
                v = window.iloc[i]
                if pd.isna(v):
                    if ignore_nan:
                        continue
                elif v in ignore_vals:
                    continue
                cnt += 1
                if cnt == k_int:
                    return v
            return np.nan
        return s.rolling(n_period).apply(kth_in_window, raw=False)

    if isinstance(data, pd.DataFrame):
        result = data.groupby(level=1).transform(kth_series)
        result.columns = [f"ts_kth_element({col}, {n_period}, {k_int})" for col in data.columns]
        return result
    else:
        res = data.transform(kth_series)
        if hasattr(data, '_selection') and data._selection is not None:
            original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
        else:
            original_columns = data.obj.columns.tolist()
        res.columns = [f"ts_kth_element({col}, {n_period}, {k_int})" for col in original_columns]
        return res

# ============================================================================
# Operator Overloading Support
# ============================================================================

def _operator_add(self, other):
    """Overload + operator for DataFrame and DataFrameGroupBy"""
    return add(self, other)

def _operator_sub(self, other):
    """Overload - operator for DataFrame and DataFrameGroupBy"""
    return subtract(self, other)

def _operator_mul(self, other):
    """Overload * operator for DataFrame and DataFrameGroupBy"""
    return multiply(self, other)

def _operator_truediv(self, other):
    """Overload / operator for DataFrame and DataFrameGroupBy"""
    return divide(self, other)

def _operator_pow(self, other):
    """Overload ** operator for DataFrame and DataFrameGroupBy"""
    return power(self, other)

def _operator_floordiv(self, other):
    """Overload // operator for DataFrame and DataFrameGroupBy"""
    return floor_divide(self, other)

def _operator_mod(self, other):
    """Overload % operator for DataFrame and DataFrameGroupBy"""
    return modulo(self, other)

def _operator_eq(self, other):
    """Overload == operator for DataFrame and DataFrameGroupBy with expression-style column names"""
    left = self.obj if isinstance(self, pd.core.groupby.DataFrameGroupBy) else self
    # Scalar or DataFrame/GroupBy right
    if isinstance(other, pd.core.groupby.DataFrameGroupBy):
        right = other.obj
    else:
        right = other

    result_frames = []
    if isinstance(right, pd.DataFrame):
        cols1 = list(left.columns)
        cols2 = list(right.columns)
        common = [c for c in cols1 if c in cols2]
        pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
        for c1, c2 in pairs:
            res_series = left[c1] == right[c2]
            result_frames.append(res_series.to_frame(name=f"{c1}=={c2}"))
    elif isinstance(right, numbers.Number):
        for c1 in list(left.columns):
            res_series = left[c1] == right
            result_frames.append(res_series.to_frame(name=f"{c1}=={right}"))
    else:
        raise TypeError("Unsupported type for ==: expected DataFrame/GroupBy or scalar")
    return pd.concat(result_frames, axis=1)

def _operator_gt(self, other):
    """Overload > operator for DataFrame and DataFrameGroupBy with expression-style column names"""
    left = self.obj if isinstance(self, pd.core.groupby.DataFrameGroupBy) else self
    if isinstance(other, pd.core.groupby.DataFrameGroupBy):
        right = other.obj
    else:
        right = other

    result_frames = []
    if isinstance(right, pd.DataFrame):
        cols1 = list(left.columns)
        cols2 = list(right.columns)
        common = [c for c in cols1 if c in cols2]
        pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
        for c1, c2 in pairs:
            res_series = left[c1] > right[c2]
            result_frames.append(res_series.to_frame(name=f"{c1}>{c2}"))
    elif isinstance(right, numbers.Number):
        for c1 in list(left.columns):
            res_series = left[c1] > right
            result_frames.append(res_series.to_frame(name=f"{c1}>{right}"))
    else:
        raise TypeError("Unsupported type for >: expected DataFrame/GroupBy or scalar")
    return pd.concat(result_frames, axis=1)

def _operator_lt(self, other):
    """Overload < operator for DataFrame and DataFrameGroupBy with expression-style column names"""
    left = self.obj if isinstance(self, pd.core.groupby.DataFrameGroupBy) else self
    if isinstance(other, pd.core.groupby.DataFrameGroupBy):
        right = other.obj
    else:
        right = other

    result_frames = []
    if isinstance(right, pd.DataFrame):
        cols1 = list(left.columns)
        cols2 = list(right.columns)
        common = [c for c in cols1 if c in cols2]
        pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
        for c1, c2 in pairs:
            res_series = left[c1] < right[c2]
            result_frames.append(res_series.to_frame(name=f"{c1}<{c2}"))
    elif isinstance(right, numbers.Number):
        for c1 in list(left.columns):
            res_series = left[c1] < right
            result_frames.append(res_series.to_frame(name=f"{c1}<{right}"))
    else:
        raise TypeError("Unsupported type for <: expected DataFrame/GroupBy or scalar")
    return pd.concat(result_frames, axis=1)

def _operator_ge(self, other):
    """Overload >= operator for DataFrame and DataFrameGroupBy with expression-style column names"""
    left = self.obj if isinstance(self, pd.core.groupby.DataFrameGroupBy) else self
    if isinstance(other, pd.core.groupby.DataFrameGroupBy):
        right = other.obj
    else:
        right = other

    result_frames = []
    if isinstance(right, pd.DataFrame):
        cols1 = list(left.columns)
        cols2 = list(right.columns)
        common = [c for c in cols1 if c in cols2]
        pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
        for c1, c2 in pairs:
            res_series = left[c1] >= right[c2]
            result_frames.append(res_series.to_frame(name=f"{c1}>={c2}"))
    elif isinstance(right, numbers.Number):
        for c1 in list(left.columns):
            res_series = left[c1] >= right
            result_frames.append(res_series.to_frame(name=f"{c1}>={right}"))
    else:
        raise TypeError("Unsupported type for >=: expected DataFrame/GroupBy or scalar")
    return pd.concat(result_frames, axis=1)

def _operator_le(self, other):
    """Overload <= operator for DataFrame and DataFrameGroupBy with expression-style column names"""
    left = self.obj if isinstance(self, pd.core.groupby.DataFrameGroupBy) else self
    if isinstance(other, pd.core.groupby.DataFrameGroupBy):
        right = other.obj
    else:
        right = other

    result_frames = []
    if isinstance(right, pd.DataFrame):
        cols1 = list(left.columns)
        cols2 = list(right.columns)
        common = [c for c in cols1 if c in cols2]
        pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
        for c1, c2 in pairs:
            res_series = left[c1] <= right[c2]
            result_frames.append(res_series.to_frame(name=f"{c1}<={c2}"))
    elif isinstance(right, numbers.Number):
        for c1 in list(left.columns):
            res_series = left[c1] <= right
            result_frames.append(res_series.to_frame(name=f"{c1}<={right}"))
    else:
        raise TypeError("Unsupported type for <=: expected DataFrame/GroupBy or scalar")
    return pd.concat(result_frames, axis=1)

def _operator_ne(self, other):
    """Overload != operator for DataFrame and DataFrameGroupBy with expression-style column names"""
    left = self.obj if isinstance(self, pd.core.groupby.DataFrameGroupBy) else self
    if isinstance(other, pd.core.groupby.DataFrameGroupBy):
        right = other.obj
    else:
        right = other

    result_frames = []
    if isinstance(right, pd.DataFrame):
        cols1 = list(left.columns)
        cols2 = list(right.columns)
        common = [c for c in cols1 if c in cols2]
        pairs = [(c, c) for c in common] if len(common) > 0 else [(cols1[0], cols2[0])]
        for c1, c2 in pairs:
            res_series = left[c1] != right[c2]
            result_frames.append(res_series.to_frame(name=f"{c1}!={c2}"))
    elif isinstance(right, numbers.Number):
        for c1 in list(left.columns):
            res_series = left[c1] != right
            result_frames.append(res_series.to_frame(name=f"{c1}!={right}"))
    else:
        raise TypeError("Unsupported type for !=: expected DataFrame/GroupBy or scalar")
    return pd.concat(result_frames, axis=1)

def _operator_and(self, other):
    """Overload & operator to perform logical AND with expression-style column names"""
    return and_(self, other)

def _operator_or(self, other):
    """Overload | operator to perform logical OR with expression-style column names"""
    return or_(self, other)

def _operator_neg(self):
    """Overload -x (negation) operator for DataFrame and DataFrameGroupBy"""
    if isinstance(self, pd.core.groupby.DataFrameGroupBy):
        # For GroupBy, apply negation to the underlying DataFrame
        df = self.obj
        result_frames = []
        for col in df.columns:
            neg_series = -df[col]
            colname = f"-{col}"
            result_frames.append(neg_series.to_frame(name=colname))
        return pd.concat(result_frames, axis=1)
    else:
        # For DataFrame, apply negation to all columns
        result_frames = []
        for col in self.columns:
            neg_series = -self[col]
            colname = f"-{col}"
            result_frames.append(neg_series.to_frame(name=colname))
        return pd.concat(result_frames, axis=1)

def _operator_pos(self):
    """Overload +x (positive) operator for DataFrame and DataFrameGroupBy"""
    if isinstance(self, pd.core.groupby.DataFrameGroupBy):
        # For GroupBy, apply positive to the underlying DataFrame
        df = self.obj
        result_frames = []
        for col in df.columns:
            pos_series = +df[col]
            colname = f"+{col}"
            result_frames.append(pos_series.to_frame(name=colname))
        return pd.concat(result_frames, axis=1)
    else:
        # For DataFrame, apply positive to all columns
        result_frames = []
        for col in self.columns:
            pos_series = +self[col]
            colname = f"+{col}"
            result_frames.append(pos_series.to_frame(name=colname))
        return pd.concat(result_frames, axis=1)

def _operator_abs(self):
    """Overload abs(x) operator for DataFrame and DataFrameGroupBy"""
    # Use the existing abs function we defined earlier
    return abs(self)

@df_scalar_broadcast_support()
def not_(data: pd.DataFrame | pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """
    逻辑取反：将输入按 0/1 逻辑取反并返回 0/1。
    - 非零且非 NaN 视为 True，结果为 0
    - 等于 0 或 NaN 视为 False，结果为 1
    - 列名规范：not(<col>)
    """
    if isinstance(data, pd.DataFrame):
        vals = ((data.values == 0) | np.isnan(data.values)).astype(int)
        res = pd.DataFrame(vals, index=data.index)
        res.columns = [f"not({col})" for col in data.columns]
        return res

    # GroupBy
    res = data.transform(lambda x: ((x == 0) | x.isna()).astype(int))
    # 还原列名
    if hasattr(data, '_selection') and data._selection is not None:
        original_columns = data._selection if isinstance(data._selection, list) else [data._selection]
    elif hasattr(data, 'obj') and hasattr(data.obj, 'columns'):
        original_columns = data.obj.columns.tolist()
    else:
        original_columns = ['data']

    if isinstance(res, pd.Series):
        col_name = original_columns[0] if len(original_columns) > 0 else (res.name if res.name else 'data')
        res = res.to_frame(f"not({col_name})")
    else:
        cols = list(res.columns)
        if len(cols) == len(original_columns):
            res.columns = [f"not({c})" for c in original_columns]
        else:
            res.columns = [f"not({c})" for c in cols]
    return res

# Monkey patch DataFrame and DataFrameGroupBy to support operator overloading
# Basic arithmetic operators
pd.DataFrame.__add__ = _operator_add
pd.DataFrame.__sub__ = _operator_sub
pd.DataFrame.__mul__ = _operator_mul
pd.DataFrame.__truediv__ = _operator_truediv
pd.DataFrame.__pow__ = _operator_pow
pd.DataFrame.__floordiv__ = _operator_floordiv
pd.DataFrame.__mod__ = _operator_mod

# Comparison operators
pd.DataFrame.__eq__ = _operator_eq
pd.DataFrame.__gt__ = _operator_gt
pd.DataFrame.__lt__ = _operator_lt
pd.DataFrame.__ge__ = _operator_ge
pd.DataFrame.__le__ = _operator_le
pd.DataFrame.__ne__ = _operator_ne
pd.DataFrame.__and__ = _operator_and
pd.DataFrame.__or__ = _operator_or

# Unary operators
pd.DataFrame.__neg__ = _operator_neg
pd.DataFrame.__pos__ = _operator_pos
pd.DataFrame.__abs__ = _operator_abs

# Apply to DataFrameGroupBy as well
pd.core.groupby.DataFrameGroupBy.__add__ = _operator_add
pd.core.groupby.DataFrameGroupBy.__sub__ = _operator_sub
pd.core.groupby.DataFrameGroupBy.__mul__ = _operator_mul
pd.core.groupby.DataFrameGroupBy.__truediv__ = _operator_truediv
pd.core.groupby.DataFrameGroupBy.__pow__ = _operator_pow
pd.core.groupby.DataFrameGroupBy.__floordiv__ = _operator_floordiv
pd.core.groupby.DataFrameGroupBy.__mod__ = _operator_mod
pd.core.groupby.DataFrameGroupBy.__eq__ = _operator_eq
pd.core.groupby.DataFrameGroupBy.__gt__ = _operator_gt
pd.core.groupby.DataFrameGroupBy.__lt__ = _operator_lt
pd.core.groupby.DataFrameGroupBy.__ge__ = _operator_ge
pd.core.groupby.DataFrameGroupBy.__le__ = _operator_le
pd.core.groupby.DataFrameGroupBy.__ne__ = _operator_ne
pd.core.groupby.DataFrameGroupBy.__and__ = _operator_and
pd.core.groupby.DataFrameGroupBy.__or__ = _operator_or
pd.core.groupby.DataFrameGroupBy.__neg__ = _operator_neg
pd.core.groupby.DataFrameGroupBy.__pos__ = _operator_pos
pd.core.groupby.DataFrameGroupBy.__abs__ = _operator_abs

