"""
================================================================================
Реализация Cross-Data-Type 1-D CNN (CDT 1-D CNN)
На основе: Wang et al. "Financial Markets Prediction with Deep Learning" (2021)
Адаптировано для: НИР — торговая стратегия на российском фондовом рынке
================================================================================

Ключевые отличия от базовой модели (models.ipynb):
1. CDT 1-D CNN: ядра сканируют по оси времени, разделяя параметры между типами данных
2. 3 сверточных + max-pooling слоя (4×32, 3×64, 2×128) вместо 2 слоёв
3. Weighted F-Score — метрика, лучше коррелирующая с финансовыми показателями
4. Moving window training (2 года обучение, 4 недели валидация, 2 недели тест)
5. Сравнение: CDT 1-D CNN w/ TIs vs w/o TIs vs базовая модель
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler

# Подавляем лишние логи TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==============================================================================
# 1. ПАРАМЕТРЫ (по статье Wang et al.)
# ==============================================================================
WINDOW_SIZE = 24        # 24 свечи = 2 часа (5-мин данные), как в статье
ALPHA = 0.55            # Параметр α для динамической разметки (из статьи)
TRAIN_RECORDS = 142_416 # ~2 года 5-мин данных (из статьи)
VAL_RECORDS = 192       # 4 недели (из статьи)
TEST_RECORDS = 96       # 2 недели (из статьи)
STEP_RECORDS = 96       # Шаг скользящего окна (из статьи)

LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.7      # Dropout из статьи
L2_DECAY = 1e-5
BATCH_SIZE = 12         # Mini-batch из статьи
EPOCHS = 15
CONFIDENCE_THRESHOLD = 0.65  # Порог уверенности для сигналов

# Weighted F-Score параметры (из статьи, Section IV-B2)
BETA1 = 0.5             # Вес ошибок 2-го уровня
BETA2 = 0.125           # Вес ошибок 3-го уровня
BETA3 = 0.125           # Вес True Flat


# ==============================================================================
# 2. ПРЕДОБРАБОТКА ДАННЫХ
# ==============================================================================
def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка рыночных данных: замена нулей, forward fill."""
    df = df.copy()
    cols_to_fix = ['Open', 'High', 'Low', 'Close']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)
    df = df.sort_values("DateTime")
    df[cols_to_fix] = df[cols_to_fix].ffill().bfill()
    df['Volume'] = df['Volume'].fillna(0)
    return df


def fill_time_gaps(df: pd.DataFrame, interval_name: str = "5min") -> pd.DataFrame:
    """Вставка пропущенных временных интервалов."""
    resample_map = {"5min": "5min", "15min": "15min", "1hour": "h", "1day": "D"}
    freq = resample_map.get(interval_name, "5min")
    df = df.copy()
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime').sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_range)
    df.index.name = 'DateTime'
    df['Close'] = df['Close'].ffill()
    for col in ['Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].fillna(df['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    return df.reset_index()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет технические индикаторы из статьи Wang et al. (Section IV-A):
    EMA, MACD Histogram, Bollinger Bands, RSI, CCI, VWAP, OBV, ADX, ADL, CMF, ROC.
    Разные временные окна для EMA, MACD, BB, ROC, RSI, CCI, CMF, ADX.
    Итого: 46 атрибутов (5 OHLCV + 41 TI), как в статье.
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # --- EMA (разные периоды) ---
    for p in [5, 10, 20, 50]:
        df[f'EMA_{p}'] = close.ewm(span=p, adjust=False).mean()

    # --- MACD Histogram (разные пары) ---
    for fast, slow, sig in [(12, 26, 9), (5, 15, 5)]:
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig, adjust=False).mean()
        df[f'MACD_Hist_{fast}_{slow}'] = macd - signal

    # --- Bollinger Bands (разные периоды) ---
    for p in [10, 20]:
        sma = close.rolling(p).mean()
        std = close.rolling(p).std()
        df[f'BB_Upper_{p}'] = sma + 2 * std
        df[f'BB_Lower_{p}'] = sma - 2 * std
        df[f'BB_Width_{p}'] = (df[f'BB_Upper_{p}'] - df[f'BB_Lower_{p}']) / sma

    # --- RSI (разные периоды) ---
    for p in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, 1e-10)
        df[f'RSI_{p}'] = 100 - (100 / (1 + rs))

    # --- CCI (разные периоды) ---
    for p in [14, 20]:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df[f'CCI_{p}'] = (tp - sma_tp) / (0.015 * mad.replace(0, 1e-10))

    # --- VWAP ---
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, 1e-10)
    df['VWAP'] = cum_tp_vol / cum_vol

    # --- OBV ---
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # --- ADX (14) ---
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10))
    df['ADX_14'] = dx.rolling(14).mean()

    # --- ADL (Accumulation Distribution Line) ---
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-10)
    df['ADL'] = (mfm * volume).cumsum()

    # --- CMF (разные периоды) ---
    for p in [10, 20]:
        mfv = mfm * volume
        df[f'CMF_{p}'] = mfv.rolling(p).sum() / volume.rolling(p).sum().replace(0, 1e-10)

    # --- ROC (разные периоды) ---
    for p in [5, 10, 20]:
        df[f'ROC_{p}'] = close.pct_change(p) * 100

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df


# ==============================================================================
# 3. ВОЛАТИЛЬНАЯ РАЗМЕТКА (из статьи, Equation 8)
# ==============================================================================
def create_labels(df: pd.DataFrame, alpha: float = ALPHA, window: int = WINDOW_SIZE) -> pd.Series:
    """
    Динамическая разметка по статье Wang et al. (Equation 8):
      up:   if c_{t+1} >= c_t * (1 + α * v_t)
      down: if c_{t+1} <= c_t * (1 - α * v_t)
      flat: otherwise
    где v_t — стандартное отклонение последних 10 точек.
    """
    close = df['Close'].values
    returns = pd.Series(close).pct_change()
    volatility = returns.rolling(window=10).std().values

    labels = np.zeros(len(close), dtype=int)  # 0 = flat

    for t in range(len(close) - 1):
        if np.isnan(volatility[t]) or volatility[t] == 0:
            continue
        c_t = close[t]
        c_next = close[t + 1]
        threshold = alpha * volatility[t]

        if c_next >= c_t * (1 + threshold):
            labels[t] = 1  # Up
        elif c_next <= c_t * (1 - threshold):
            labels[t] = 2  # Down
        # else: flat (0)

    return labels


# ==============================================================================
# 4. СОЗДАНИЕ ТЕНЗОРОВ (CDT формат из статьи)
# ==============================================================================
def create_tensors_cdt(df: pd.DataFrame, feature_cols: list,
                       window_size: int = WINDOW_SIZE,
                       alpha: float = ALPHA):
    """
    Формирует 2D-фреймы для CDT 1-D CNN:
    - x-axis: время (window_size точек)
    - y-axis: типы данных (OHLCV или OHLCV + TIs)

    Возвращает: X (N, window_size, n_features), y, dates, prices
    """
    df = df.copy().sort_values("DateTime").reset_index(drop=True)

    # Разметка
    labels = create_labels(df, alpha=alpha, window=window_size)
    df['Target'] = labels

    data_x = df[feature_cols].values
    data_y = df['Target'].values
    dates = df['DateTime'].values
    prices = df['Close'].values

    X, y, out_dates, out_prices = [], [], [], []

    for i in range(window_size, len(df) - 1):
        window = data_x[i - window_size:i]

        # Z-score нормализация окна (как в вашей реализации)
        scaler = StandardScaler()
        norm_window = scaler.fit_transform(window)

        X.append(norm_window)
        y.append(data_y[i])
        out_dates.append(dates[i])
        out_prices.append(prices[i])

    return np.array(X), np.array(y), np.array(out_dates), np.array(out_prices)


# ==============================================================================
# 5. АРХИТЕКТУРА CDT 1-D CNN (из статьи, Section III-A, Figure 1)
# ==============================================================================
def _get_adam(learning_rate: float):
    """
    Возвращает оптимизатор Adam, совместимый с текущей платформой.
    На Mac M1/M2/M3 стандартный tf.keras.optimizers.Adam (v2.11+) работает
    медленно из-за отсутствия оптимизаций для Apple Silicon.
    legacy.Adam использует старую реализацию, которая быстрее на этих чипах.
    """
    try:
        opt = optimizers.legacy.Adam(learning_rate=learning_rate)
        return opt
    except AttributeError:
        # legacy недоступен (TF < 2.11 или будущие версии) — используем стандартный
        return optimizers.Adam(learning_rate=learning_rate)



def build_cdt_1d_cnn(n_features: int, window_size: int = WINDOW_SIZE,
                     n_classes: int = 3) -> tf.keras.Model:
    """
    Cross-Data-Type 1-D CNN из статьи Wang et al.

    Архитектура (Section V):
    - 3 CDT 1-D Conv + MaxPool слоя
    - Ядра: 4×32, 3×64, 2×128
    - Max-pool стrides: 4, 3, 2
    - 2 FC слоя: 1000 и 500 юнитов
    - Softmax выход

    CDT-особенность: 1D-свёртки сканируют по оси времени (axis=1),
    параметры разделяются между типами данных (OHLCV).
    Входные данные: (batch, window_size, n_features)
    """
    inputs = layers.Input(shape=(window_size, n_features), name='input_frame')

    # === CDT 1-D Convolution Layer 1 ===
    # kernel_size=4, 32 фильтров
    x = layers.Conv1D(
        filters=32, kernel_size=4, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY),
        name='cdt_conv1'
    )(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling1D(pool_size=4, strides=4, name='pool1')(x)

    # === CDT 1-D Convolution Layer 2 ===
    # kernel_size=3, 64 фильтра
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY),
        name='cdt_conv2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=3, name='pool2')(x)

    # === CDT 1-D Convolution Layer 3 ===
    # kernel_size=2, 128 фильтров
    x = layers.Conv1D(
        filters=128, kernel_size=2, padding='same', activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY),
        name='cdt_conv3'
    )(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='pool3')(x)

    # === Fully Connected Layers ===
    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(1000, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY),
                     name='fc1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout1')(x)

    x = layers.Dense(500, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY),
                     name='fc2')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout2')(x)

    # === Output ===
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CDT_1D_CNN')

    model.compile(
        optimizer=_get_adam(LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_baseline_cnn(n_features: int, window_size: int = WINDOW_SIZE,
                       n_classes: int = 3) -> tf.keras.Model:
    """
    Базовая модель (ваша текущая архитектура из models.ipynb) для сравнения.
    """
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                      input_shape=(window_size, n_features)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ], name='Baseline_1D_CNN')

    model.compile(
        optimizer=_get_adam(0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==============================================================================
# 6. WEIGHTED F-SCORE (из статьи, Section IV-B2, Equations 12-16)
# ==============================================================================
def weighted_f_score(y_true: np.ndarray, y_pred: np.ndarray,
                     beta1: float = BETA1, beta2: float = BETA2,
                     beta3: float = BETA3) -> dict:
    """
    Weighted F-Score из статьи Wang et al. (Equations 12-16).

    Классы: 0=Flat, 1=Up, 2=Down

    Типы ошибок (по убыванию критичности):
    - 1st: предсказание Up, реально Down (и наоборот) — торговля в неверном направлении
    - 2nd: предсказание Up/Down, реально Flat — лишняя сделка (slippage + комиссия)
    - 3rd: предсказание Flat, реально Up/Down — упущенная возможность

    Returns: dict с WFS, Accuracy, ошибками по типам
    """
    n = len(y_true)
    assert n == len(y_pred)

    # True counts
    N_tu = np.sum((y_true == 1) & (y_pred == 1))  # True Up
    N_td = np.sum((y_true == 2) & (y_pred == 2))  # True Down
    N_tf = np.sum((y_true == 0) & (y_pred == 0))  # True Flat

    # 1st type errors (worst): opposite direction
    E_wutd = np.sum((y_pred == 1) & (y_true == 2))  # Predicted Up, True Down
    E_wdtu = np.sum((y_pred == 2) & (y_true == 1))  # Predicted Down, True Up
    E_1st = E_wutd + E_wdtu

    # 2nd type errors: predicted directional, true flat
    E_wutf = np.sum((y_pred == 1) & (y_true == 0))  # Predicted Up, True Flat
    E_wdtf = np.sum((y_pred == 2) & (y_true == 0))  # Predicted Down, True Flat
    E_2nd = E_wutf + E_wdtf

    # 3rd type errors (least bad): predicted flat, true directional
    E_wftu = np.sum((y_pred == 0) & (y_true == 1))  # Predicted Flat, True Up
    E_wftd = np.sum((y_pred == 0) & (y_true == 2))  # Predicted Flat, True Down
    E_3rd = E_wftu + E_wftd

    # Weighted True Positive (Eq. 12)
    N_tp = N_tu + N_td + beta3**2 * N_tf

    # Weighted F-Score (Eq. 16)
    numerator = (1 + beta1**2 + beta2**2) * N_tp
    denominator = numerator + E_1st + beta1**2 * E_2nd + beta2**2 * E_3rd

    wfs = numerator / denominator if denominator > 0 else 0.0

    # Обычная accuracy для сравнения
    accuracy = np.sum(y_true == y_pred) / n if n > 0 else 0.0

    return {
        'WFS': wfs,
        'Accuracy': accuracy,
        'N_true_up': int(N_tu),
        'N_true_down': int(N_td),
        'N_true_flat': int(N_tf),
        'E_1st_opposite': int(E_1st),
        'E_2nd_unnecessary': int(E_2nd),
        'E_3rd_missed': int(E_3rd),
        'Total': n
    }


# ==============================================================================
# 7. БЭКТЕСТ (по статье Wang et al. с moving windows + breakeven из вашего проекта)
# ==============================================================================
def run_moving_window_backtest(df: pd.DataFrame,
                               feature_cols: list,
                               model_builder,
                               model_name: str = "CDT 1-D CNN",
                               confidence_threshold: float = CONFIDENCE_THRESHOLD,
                               alpha: float = ALPHA,
                               commission: float = 0.0003,
                               be_trigger_pct: float = 1.0):
    """
    Бэктест по методологии Wang et al. со скользящими окнами:
    - Train: ~2 года
    - Validation: 4 недели
    - Test: 2 недели
    - Шаг: 2 недели

    + breakeven логика из вашего проекта (перенос SL в безубыток при +be_trigger_pct%)
    """
    print(f"\n{'='*70}")
    print(f"  БЭКТЕСТ: {model_name}")
    print(f"  Признаки: {len(feature_cols)} | Окно: {WINDOW_SIZE} | α={alpha}")
    print(f"{'='*70}")

    # Создание тензоров
    X_all, y_all, dates_all, prices_all = create_tensors_cdt(
        df, feature_cols, window_size=WINDOW_SIZE, alpha=alpha
    )

    n_features = X_all.shape[2]
    print(f"Всего точек: {len(X_all)} | Форма входа: {X_all.shape}")

    # Распределение классов
    unique, counts = np.unique(y_all, return_counts=True)
    for u, c in zip(unique, counts):
        lbl = {0: 'Flat', 1: 'Up', 2: 'Down'}[u]
        print(f"  Класс {lbl}: {c} ({c/len(y_all)*100:.1f}%)")

    # === Moving window training ===
    all_preds = []
    all_actuals = []
    all_dates = []
    all_prices = []

    # Определяем размеры окон
    # Для российских данных масштабируем пропорционально доступным данным
    total = len(X_all)
    train_size = min(TRAIN_RECORDS, int(total * 0.6))
    val_size = min(VAL_RECORDS, int(total * 0.05))
    test_size = min(TEST_RECORDS, int(total * 0.025))
    step_size = test_size

    print(f"\nMoving Window: train={train_size}, val={val_size}, "
          f"test={test_size}, step={step_size}")

    window_start = 0
    session = 0

    while window_start + train_size + val_size + test_size <= total:
        train_end = window_start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size

        X_train = X_all[window_start:train_end]
        y_train = y_all[window_start:train_end]
        X_val = X_all[train_end:val_end]
        y_val = y_all[train_end:val_end]
        X_test = X_all[val_end:test_end]
        y_test = y_all[val_end:test_end]

        if len(X_test) == 0:
            break

        # Строим модель заново (как в статье: each training session starts from scratch)
        model = model_builder(n_features=n_features, window_size=WINDOW_SIZE)

        # Early stopping по валидационной выборке
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        # Предсказание на тесте с фильтрацией уверенности
        probs = model.predict(X_test, verbose=0)
        preds = []
        for p in probs:
            if p[1] > confidence_threshold:
                preds.append(1)  # Up
            elif p[2] > confidence_threshold:
                preds.append(2)  # Down
            else:
                preds.append(0)  # Flat

        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates.extend(dates_all[val_end:test_end])
        all_prices.extend(prices_all[val_end:test_end])

        session += 1
        window_start += step_size

        if session % 10 == 0:
            print(f"  Сессия {session}: обработано {len(all_preds)} точек...")

        # Очистка памяти
        del model
        tf.keras.backend.clear_session()

    print(f"  Всего сессий: {session}, предсказаний: {len(all_preds)}")

    if len(all_preds) == 0:
        print("  ОШИБКА: нет предсказаний!")
        return None

    # === Метрики ===
    y_pred_arr = np.array(all_preds)
    y_true_arr = np.array(all_actuals)
    prices_arr = np.array(all_prices)
    dates_arr = np.array(all_dates)

    # Weighted F-Score
    wfs_result = weighted_f_score(y_true_arr, y_pred_arr)
    print(f"\n  Weighted F-Score: {wfs_result['WFS']*100:.1f}%")
    print(f"  Accuracy:         {wfs_result['Accuracy']*100:.1f}%")
    print(f"  Ошибки 1-го типа (направление): {wfs_result['E_1st_opposite']}")
    print(f"  Ошибки 2-го типа (лишние):      {wfs_result['E_2nd_unnecessary']}")
    print(f"  Ошибки 3-го типа (упущенные):   {wfs_result['E_3rd_missed']}")

    # === Бэктест с breakeven ===
    trades = []
    equity_curve = [100_000]  # $100,000 начальный капитал (как в статье)
    in_position = False
    entry_price = 0.0
    entry_date = None
    position_dir = 0  # 1=long, -1=short
    is_breakeven = False

    for i in range(len(y_pred_arr)):
        signal = y_pred_arr[i]
        price = prices_arr[i]

        if in_position:
            # Проверка breakeven
            if position_dir == 1:
                unrealized_pct = (price - entry_price) / entry_price * 100
            else:
                unrealized_pct = (entry_price - price) / entry_price * 100

            if unrealized_pct >= be_trigger_pct and not is_breakeven:
                is_breakeven = True

            # Условие закрытия: разворот сигнала
            should_close = False
            if position_dir == 1 and signal == 2:
                should_close = True
            elif position_dir == -1 and signal == 1:
                should_close = True
            elif is_breakeven and unrealized_pct <= 0:
                should_close = True  # Breakeven stop

            if should_close:
                if position_dir == 1:
                    pnl_pct = (price - entry_price) / entry_price - commission * 2
                else:
                    pnl_pct = (entry_price - price) / entry_price - commission * 2

                equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                trades.append({
                    'Entry': entry_date,
                    'Exit': dates_arr[i],
                    'Direction': 'Long' if position_dir == 1 else 'Short',
                    'PnL_pct': pnl_pct * 100
                })
                in_position = False
                is_breakeven = False

                # Сразу входим в противоположную позицию
                if signal == 1:
                    in_position = True
                    entry_price = price
                    entry_date = dates_arr[i]
                    position_dir = 1
                    is_breakeven = False
                elif signal == 2:
                    in_position = True
                    entry_price = price
                    entry_date = dates_arr[i]
                    position_dir = -1
                    is_breakeven = False
            else:
                equity_curve.append(equity_curve[-1])
        else:
            # Открытие позиции
            if signal == 1:
                in_position = True
                entry_price = price
                entry_date = dates_arr[i]
                position_dir = 1
                is_breakeven = False
            elif signal == 2:
                in_position = True
                entry_price = price
                entry_date = dates_arr[i]
                position_dir = -1
                is_breakeven = False
            equity_curve.append(equity_curve[-1])

    # === Расчёт финансовых метрик ===
    equity_arr = np.array(equity_curve)
    total_return = (equity_arr[-1] / equity_arr[0] - 1) * 100

    # Определяем период теста
    if len(dates_arr) > 0:
        test_days = (pd.Timestamp(dates_arr[-1]) - pd.Timestamp(dates_arr[0])).days
        test_years = max(test_days / 365.25, 0.01)
    else:
        test_years = 1.0

    annual_return = ((equity_arr[-1] / equity_arr[0]) ** (1 / test_years) - 1) * 100

    # Sharpe Ratio (из статьи)
    daily_returns = np.diff(equity_arr) / equity_arr[:-1]
    daily_returns = daily_returns[daily_returns != 0]
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252 * 78)
    else:
        sharpe = 0.0

    # Profit Factor и другие метрики
    if trades:
        trades_df = pd.DataFrame(trades)
        pnls = trades_df['PnL_pct'].values
        n_trades = len(pnls)
        win_rate = np.sum(pnls > 0) / n_trades * 100

        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = abs(np.sum(pnls[pnls < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean(pnls[pnls > 0]) if np.any(pnls > 0) else 0
        avg_loss = abs(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # MDE (статистическая значимость)
        se = np.std(pnls) / np.sqrt(n_trades) if n_trades > 1 else 0
        mde = 2.8 * se
        avg_trade = np.mean(pnls)
        stat_sig = abs(avg_trade) > mde
    else:
        n_trades = win_rate = profit_factor = rr_ratio = sharpe = 0
        avg_trade = mde = 0
        stat_sig = False

    # === Вывод результатов ===
    print(f"\n{'═'*70}")
    print(f"  РЕЗУЛЬТАТЫ: {model_name}")
    print(f"{'═'*70}")
    print(f"  Период теста:           {test_days} дней ≈ {test_years:.1f} лет")
    print(f"  Всего сделок:           {n_trades}")
    print(f"  Win Rate:               {win_rate:.1f}%")
    print(f"  Общая доходность:       {total_return:.1f}%")
    print(f"  Средняя годовая доход.: {annual_return:.1f}%")
    print(f"  Средняя сделка:         {avg_trade:.2f}%")
    print(f"  Profit Factor:          {profit_factor:.2f}")
    print(f"  Risk/Reward Ratio:      {rr_ratio:.2f}")
    print(f"  Sharpe Ratio:           {sharpe:.2f}")
    print(f"  Weighted F-Score:       {wfs_result['WFS']*100:.1f}%")
    print(f"  Accuracy:               {wfs_result['Accuracy']*100:.1f}%")
    print(f"  MDE:                    {mde:.2f}%")
    print(f"  Стат. значимость:       {'ДА' if stat_sig else 'НЕТ (шум)'}")
    print(f"{'═'*70}")

    return {
        'model_name': model_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'wfs': wfs_result['WFS'],
        'accuracy': wfs_result['Accuracy'],
        'profit_factor': profit_factor,
        'rr_ratio': rr_ratio,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'stat_sig': stat_sig,
        'equity_curve': equity_arr,
        'dates': dates_arr,
        'trades': trades if trades else [],
        'wfs_details': wfs_result,
    }


# ==============================================================================
# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (стиль из статьи — Fig. 2, 3, 4, 5)
# ==============================================================================
def plot_results_comparison(results_list: list, ticker: str = ""):
    """
    Визуализация в стиле статьи Wang et al. (Figures 2-5):
    1. Equity curves для всех моделей
    2. Сравнительная таблица AAR и Sharpe
    3. Cross-correlation WFS vs AAR/SR
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Сравнение моделей — {ticker}', fontsize=16, fontweight='bold')

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    # --- 1. Equity Curves (аналог Fig. 2a) ---
    ax = axes[0, 0]
    for i, res in enumerate(results_list):
        eq = res['equity_curve']
        cum_ret = (eq / eq[0] - 1) * 100
        ax.plot(cum_ret, label=res['model_name'], color=colors[i % len(colors)], linewidth=1.5)
    ax.set_title('Кумулятивная доходность (%)', fontweight='bold')
    ax.set_xlabel('Торговые сигналы')
    ax.set_ylabel('Cumulative Return, %')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # --- 2. Average Annual Return (аналог Fig. 3) ---
    ax = axes[0, 1]
    names = [r['model_name'] for r in results_list]
    aars = [r['annual_return'] for r in results_list]
    bars = ax.bar(range(len(names)), aars, color=colors[:len(names)], alpha=0.8)
    ax.set_title('Средняя годовая доходность, %', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('AAR, %')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, aars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # --- 3. Sharpe Ratio (аналог Fig. 4) ---
    ax = axes[1, 0]
    sharpes = [r['sharpe'] for r in results_list]
    bars = ax.bar(range(len(names)), sharpes, color=colors[:len(names)], alpha=0.8)
    ax.set_title('Коэффициент Шарпа', fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

    # --- 4. WFS vs Accuracy (аналог Fig. 5) ---
    ax = axes[1, 1]
    wfs_vals = [r['wfs'] * 100 for r in results_list]
    acc_vals = [r['accuracy'] * 100 for r in results_list]
    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, wfs_vals, width, label='Weighted F-Score', color='#2196F3', alpha=0.8)
    ax.bar(x_pos + width/2, acc_vals, width, label='Accuracy', color='#FF9800', alpha=0.8)
    ax.set_title('WFS vs Accuracy', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('%')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('cdt_1d_cnn_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Графики сохранены: cdt_1d_cnn_comparison.png")


def print_summary_table(results_list: list):
    """Сводная таблица (аналог Table I из статьи)."""
    print(f"\n{'='*90}")
    print(f"{'СВОДНАЯ ТАБЛИЦА':^90}")
    print(f"{'='*90}")
    header = f"{'Модель':<30} {'WFS':>6} {'ACC':>6} {'AAR':>8} {'Sharpe':>8} {'PF':>6} {'WR':>6} {'Сделки':>7}"
    print(header)
    print(f"{'-'*90}")
    for r in results_list:
        row = (f"{r['model_name']:<30} "
               f"{r['wfs']*100:>5.1f}% "
               f"{r['accuracy']*100:>5.1f}% "
               f"{r['annual_return']:>7.1f}% "
               f"{r['sharpe']:>8.2f} "
               f"{r['profit_factor']:>5.2f} "
               f"{r['win_rate']:>5.1f}% "
               f"{r['n_trades']:>7d}")
        print(row)
    print(f"{'='*90}")

    # Выделяем лучшую модель
    best = max(results_list, key=lambda x: x['wfs'])
    print(f"\n  Лучшая модель по WFS: {best['model_name']} ({best['wfs']*100:.1f}%)")
    best_sr = max(results_list, key=lambda x: x['sharpe'])
    print(f"  Лучшая модель по Sharpe: {best_sr['model_name']} ({best_sr['sharpe']:.2f})")


# ==============================================================================
# 9. ГЛАВНЫЙ ПАЙПЛАЙН
# ==============================================================================
def main(data_path: str = None, ticker: str = "YDEX"):
    """
    Главный пайплайн: сравнение 3 моделей из статьи Wang et al.
    на данных вашего проекта.

    1. CDT 1-D CNN без TI (только OHLCV) — основная модель из статьи
    2. CDT 1-D CNN с TI (OHLCV + технические индикаторы)
    3. Baseline 1-D CNN без TI (ваша текущая архитектура)
    """
    print("=" * 70)
    print("  CDT 1-D CNN — реализация по Wang et al. (2021)")
    print("  Адаптировано для российского фондового рынка")
    print("=" * 70)

    # === Загрузка данных ===
    if data_path is None:
        data_path = f'data/{ticker}_5min.parquet'

    print(f"\nЗагрузка данных: {data_path}")
    df = pd.read_parquet(data_path)
    df = df[df['DateTime'] >= '2020-01-01']
    df = df[df['DateTime'] < '2026-01-01']
    df = fill_time_gaps(df)
    df = clean_market_data(df)
    print(f"Данные: {len(df)} строк, {df['DateTime'].min()} — {df['DateTime'].max()}")

    # === Подготовка версий данных ===
    # Базовые OHLCV признаки
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']

    # С техническими индикаторами
    df_with_ti = add_technical_indicators(df.copy())
    ti_features = [c for c in df_with_ti.columns if c not in ['DateTime', 'Target']]
    print(f"\nПризнаки без TI: {len(base_features)}")
    print(f"Признаки с TI:   {len(ti_features)}")

    results_all = []

    # === Модель 1: CDT 1-D CNN без TI (как в статье — лучший результат) ===
    res1 = run_moving_window_backtest(
        df=df.copy(),
        feature_cols=base_features,
        model_builder=build_cdt_1d_cnn,
        model_name="CDT 1-D CNN w/o TIs",
        alpha=ALPHA
    )
    if res1:
        results_all.append(res1)

    # === Модель 2: CDT 1-D CNN с TI ===
    res2 = run_moving_window_backtest(
        df=df_with_ti.copy(),
        feature_cols=ti_features,
        model_builder=build_cdt_1d_cnn,
        model_name="CDT 1-D CNN w/ TIs",
        alpha=ALPHA
    )
    if res2:
        results_all.append(res2)

    # === Модель 3: Baseline 1-D CNN (ваша текущая модель) ===
    res3 = run_moving_window_backtest(
        df=df.copy(),
        feature_cols=base_features,
        model_builder=build_baseline_cnn,
        model_name="Baseline 1-D CNN",
        alpha=ALPHA
    )
    if res3:
        results_all.append(res3)

    # === Сводные результаты ===
    if results_all:
        print_summary_table(results_all)
        plot_results_comparison(results_all, ticker=ticker)

    return results_all


# ==============================================================================
# 10. ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    # Можно запустить для любого тикера:
    # results = main(ticker="SBER")
    # results = main(ticker="LKOH")
    # results = main(ticker="GLDRUB_TOM")

    results = main(ticker="YDEX")
