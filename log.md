兼容输入，dataframe输出
instrument 索引
datetime 索引
多列
数组输入
改变原数据形状再去和处理后数据相减
更改命名格式
对这几个函数也作类似刚才的修改：更改列名、确保不出现列名不匹配错误、确保返回值为dataframe、移除强制索引名设置、修复条件判断等
不允许任何series输入输出出现，只允许作为中间变量格式
创建一个大的脚本对所有cs函数做统一测试，要求测试dataframe和dataframegroupby两种不同类型数据输入、测试单列和多列数据、要显示输入输出数据前十行后十行，显示输入输出数据类型，脚本输出最终导出成txt

df与标量（int、double、float）不兼容问题未解决

# ==================== 全局数值运算精度配置 ====================

# 默认精度配置常量
DEFAULT_FLOAT_PRECISION = 8  # 浮点数显示精度
DEFAULT_NUMPY_PRECISION = 8  # NumPy数组打印精度
DEFAULT_DECIMAL_PLACES = 6   # 计算结果保留小数位数
DEFAULT_RTOL = 1e-9         # 相对容差
DEFAULT_ATOL = 1e-12        # 绝对容差
DEFAULT_EPSILON = 1e-10     # 防止除零的最小值

# 全局精度配置字典
PRECISION_CONFIG = {
    'float_precision': DEFAULT_FLOAT_PRECISION,
    'numpy_precision': DEFAULT_NUMPY_PRECISION,
    'decimal_places': DEFAULT_DECIMAL_PLACES,
    'rtol': DEFAULT_RTOL,
    'atol': DEFAULT_ATOL,
    'epsilon': DEFAULT_EPSILON,
    'handle_inf': True,      # 是否处理无穷大值
    'handle_nan': True,      # 是否处理NaN值
    'round_result': False,   # 是否对最终结果进行四舍五入
}


def set_global_precision(float_precision=None, numpy_precision=None, decimal_places=None,
                        rtol=None, atol=None, epsilon=None, handle_inf=None, 
                        handle_nan=None, round_result=None):
    """
    设置全局数值运算精度配置
    
    Args:
        float_precision: 浮点数显示精度，影响pandas显示
        numpy_precision: NumPy数组打印精度
        decimal_places: 计算结果保留小数位数
        rtol: 相对容差，用于数值比较
        atol: 绝对容差，用于数值比较
        epsilon: 防止除零的最小值
        handle_inf: 是否处理无穷大值
        handle_nan: 是否处理NaN值
        round_result: 是否对最终结果进行四舍五入
    """
    global PRECISION_CONFIG
    
    if float_precision is not None:
        PRECISION_CONFIG['float_precision'] = float_precision
        pd.set_option('display.precision', float_precision)
    
    if numpy_precision is not None:
        PRECISION_CONFIG['numpy_precision'] = numpy_precision
        np.set_printoptions(precision=numpy_precision)
    
    if decimal_places is not None:
        PRECISION_CONFIG['decimal_places'] = decimal_places
    
    if rtol is not None:
        PRECISION_CONFIG['rtol'] = rtol
    
    if atol is not None:
        PRECISION_CONFIG['atol'] = atol
    
    if epsilon is not None:
        PRECISION_CONFIG['epsilon'] = epsilon
    
    if handle_inf is not None:
        PRECISION_CONFIG['handle_inf'] = handle_inf
    
    if handle_nan is not None:
        PRECISION_CONFIG['handle_nan'] = handle_nan
    
    if round_result is not None:
        PRECISION_CONFIG['round_result'] = round_result


def get_precision_config():
    """获取当前精度配置"""
    return PRECISION_CONFIG.copy()


def apply_precision_to_result(result):
    """
    对计算结果应用精度设置
    
    Args:
        result: 计算结果（DataFrame或Series）
    
    Returns:
        应用精度设置后的结果
    """
    if not isinstance(result, (pd.DataFrame, pd.Series)):
        return result
    
    # 处理无穷大值
    if PRECISION_CONFIG['handle_inf']:
        result = result.replace([np.inf, -np.inf], np.nan)
    
    # 处理NaN值（可选择填充策略）
    if PRECISION_CONFIG['handle_nan']:
        # 这里可以根据需要选择不同的NaN处理策略
        # 目前保持NaN不变，用户可以根据具体需求修改
        pass
    
    # 四舍五入结果
    if PRECISION_CONFIG['round_result']:
        result = result.round(PRECISION_CONFIG['decimal_places'])
    
    return result


def safe_divide(numerator, denominator, fill_value=np.nan):
    """
    安全除法，避免除零错误。
    
    默认行为：当分母接近零（|den| < epsilon）时，将结果设置为NaN；
    如需使用一个固定替代值（例如epsilon或其他常数），可通过传入fill_value覆盖该行为。
    
    Args:
        numerator: 分子（pandas或numpy对象/标量）
        denominator: 分母（pandas或numpy对象/标量）
        fill_value: 分母接近零时的替代值，默认NaN（使用pd.isna检测）
    
    Returns:
        与输入类型相匹配的除法结果
    """
    epsilon = PRECISION_CONFIG['epsilon']

    # 统一判断fill_value是否为“缺失值”
    use_nan = False
    try:
        use_nan = pd.isna(fill_value)
    except Exception:
        # 如果pd.isna不可用或抛错，退回到numpy判断
        try:
            use_nan = np.isnan(fill_value)
        except Exception:
            use_nan = False

    # 构造安全分母
    safe_denominator = denominator.copy() if hasattr(denominator, 'copy') else denominator

    if hasattr(pd, 'Series') and isinstance(safe_denominator, (pd.Series, pd.DataFrame)):
        mask = safe_denominator.abs() < epsilon if isinstance(safe_denominator, pd.Series) else safe_denominator.abs() < epsilon
        replacement = np.nan if use_nan else fill_value
        safe_denominator = safe_denominator.where(~mask, replacement)
    else:
        # numpy数组或标量
        try:
            arr = np.asarray(safe_denominator)
            replacement = np.nan if use_nan else fill_value
            arr = np.where(np.abs(arr) < epsilon, replacement, arr)
            safe_denominator = arr
        except Exception:
            # 标量情况
            if np.abs(safe_denominator) < epsilon:
                safe_denominator = np.nan if use_nan else fill_value

    return numerator / safe_denominator


def is_close(a, b, rtol=None, atol=None):
    """
    使用配置的容差进行数值比较
    
    Args:
        a, b: 要比较的数值
        rtol: 相对容差，如果为None则使用全局配置
        atol: 绝对容差，如果为None则使用全局配置
    
    Returns:
        布尔值，表示两个数值是否接近
    """
    if rtol is None:
        rtol = PRECISION_CONFIG['rtol']
    if atol is None:
        atol = PRECISION_CONFIG['atol']
    
    return np.allclose(a, b, rtol=rtol, atol=atol)


# 配置numpy错误处理
def configure_numpy_precision():
    """配置numpy的精度和错误处理设置"""
    # 设置numpy打印选项
    np.set_printoptions(
        precision=PRECISION_CONFIG['numpy_precision'],
        suppress=True,  # 抑制科学计数法显示小数
        floatmode='fixed',  # 固定小数点格式
        threshold=1000,  # 数组显示阈值
        linewidth=120   # 每行字符数
    )
    
    # 设置numpy错误处理
    np.seterr(
        divide='ignore',    # 除零时忽略警告（返回inf）
        over='warn',        # 溢出时发出警告
        under='ignore',     # 下溢时忽略
        invalid='ignore'    # 无效操作时忽略（如sqrt(-1)）
    )


def configure_pandas_precision():
    """配置pandas的显示精度和数值处理选项"""
    # 设置pandas显示选项
    pd.set_option('display.precision', PRECISION_CONFIG['float_precision'])
    pd.set_option('display.float_format', f'{{:.{PRECISION_CONFIG["decimal_places"]}f}}'.format)
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)        # 自动调整宽度
    pd.set_option('display.max_colwidth', 100)  # 列最大宽度
    
    # 设置pandas计算选项
    pd.set_option('mode.use_inf_as_na', PRECISION_CONFIG['handle_inf'])  # 将inf视为NaN


# 初始化默认精度设置
set_global_precision(
    float_precision=DEFAULT_FLOAT_PRECISION,
    numpy_precision=DEFAULT_NUMPY_PRECISION
)

# 应用numpy和pandas精度配置
configure_numpy_precision()
configure_pandas_precision()

# ==================== 精度配置结束 ====================

