# Отчёт — тестирование и мониторинг ML-сервиса

## 1. Доработка кода: обработка ошибок

Добавлена обработка следующих ситуаций:

| Ситуация | HTTP-код | Где обрабатывается |
|---|---|---|
| Модель ещё не загружена | 503 | `POST /predict` |
| Некорректный тип поля (например, `age: "abc"`) | 422 | Pydantic автоматически |
| Ошибка препроцессинга признаков | 422 | `POST /predict`, блок try/except |
| Отсутствуют обязательные для модели признаки | 422 | `POST /predict`, проверка `missing` |
| Невалидный или несуществующий `run_id` | 404 | `POST /updateModel`, перехват `MlflowException` |
| Пустой `run_id` | 422 | `POST /updateModel`, явная проверка |
| Ошибка инференса модели | 500 | `POST /predict`, блок try/except |
| Любое необработанное исключение | 500 | Глобальный `exception_handler` |

Все ошибки логируются через стандартный Python `logging`. Ошибки 5XX дополнительно инкрементируют счётчик `http_errors_total` в Prometheus.

---

## 2. Тесты

### `tests/test_preprocessing.py` — функции предобработки

- `to_dataframe` возвращает `pd.DataFrame` с одной строкой
- Без фильтрации возвращаются все колонки из `FEATURE_COLUMNS`
- `needed_columns` корректно фильтрует нужные признаки
- Значения признаков из запроса попадают в DataFrame без изменений
- Поля с точками в имени (`capital.gain`, `native.country`) корректно резолвятся через pydantic-алиасы
- `None`-значения проходят насквозь
- Неизвестные колонки в `needed_columns` молча игнорируются

### `tests/test_schemas.py` — валидация схем

- `PredictRequest`: все поля опциональны, принимает и алиасы (`capital.gain`), и python-имена (`capital_gain`), некорректный тип вызывает `ValidationError`
- `PredictResponse`: поля `prediction` (int) и `probability` (float) обязательны
- `UpdateModelRequest`: `run_id` обязателен, отсутствие вызывает `ValidationError`

### `tests/test_handlers.py` — логика хэндлеров (mocked model)

- `GET /health` возвращает `{"status": "ok", "run_id": ...}`
- `POST /predict` с валидным запросом → 200, содержит `prediction` и `probability`
- Вероятность > 0.5 → `prediction=1`, < 0.5 → `prediction=0`
- Вероятность всегда в диапазоне [0, 1]
- Некорректный тип поля → 422
- Модель не загружена → 503
- Ошибка инференса → 500
- `POST /updateModel` с валидным `run_id` → 200
- Пустой `run_id` → 422
- Отсутствующий `run_id` в теле → 422
- Несуществующий `run_id` в MLflow → 404

### `tests/test_integration.py` — полный сервис с реальной моделью

Запускаются при наличии переменной окружения `TEST_RUN_ID`, иначе пропускаются.

- Сервис поднимается и `GET /health` возвращает `ok`
- `POST /predict` с валидным payload → 200, корректная структура ответа
- Вероятность в [0, 1], предсказание в {0, 1}
- Обновление модели на тот же `run_id` → 200
- Обновление на несуществующий `run_id` → 404
- Пустой `run_id` → 422

---

## 3. Метрики Prometheus (`GET /metrics`)

### Технические метрики сервиса

| Метрика | Тип | Описание |
|---|---|---|
| `http_requests_total` | Counter | Число запросов по `method`, `endpoint`, `status_code` |
| `http_request_duration_seconds` | Histogram | Время ответа по endpoint (перцентили 75/90/95/99/99.9) |
| `http_errors_total` | Counter | Число 5XX ошибок по endpoint |

### Метрики входных данных

| Метрика | Тип | Описание |
|---|---|---|
| `preprocessing_duration_seconds` | Histogram | Время препроцессинга входных данных |
| `feature_value` | Histogram | Распределение значений числовых признаков (`capital.gain`) |

### Метрики модели

| Метрика | Тип | Описание |
|---|---|---|
| `model_inference_duration_seconds` | Histogram | Время инференса модели (перцентили 75/90/95/99/99.9) |
| `model_predictions_total` | Counter | Число предсказаний по классу (0 / 1) |
| `model_prediction_probability` | Histogram | Распределение вероятностей предсказаний |

### Метрики состояния модели

| Метрика | Тип | Описание |
|---|---|---|
| `model_updates_total` | Counter | Число обновлений модели по статусу (`success` / `failure`) |
| `current_model_info` | Info | Текущий `run_id`, список и количество признаков |

По всем перечисленным метрикам создан дашборд в Grafana. Перцентили времени ответа и `current_model_info` не вынесены в отдельные панели — первые покрыты общей панелью HTTP Response Time, вторая является метрикой типа `Info` и не визуализируется в виде графика.

---

## 4. Алертинг

Алерты настроены в Grafana через **email contact point**.

**Почта для алертов:** `alertmeplease@atomicmail.io`  
**Пароль:** `EBJ-adf-rQV-9DP`

| Алерт | Условие | Метрика |
|---|---|---|
| `High 5XX error rate` | > 0.1 rps | `rate(http_errors_total[5m])` |
| `Slow model inference` | p95 > 1s | `histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))` |
| `Slow HTTP response` | p95 > 2s | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[5m]))` |
| `Model probability drift` | медиана вне [0.2, 0.8] | `histogram_quantile(0.5, rate(model_prediction_probability_bucket[10m]))` |
| `Prediction class imbalance` | доля класса 1 вне [0.1, 0.9] | `rate(model_predictions_total{prediction="1"}[10m]) / rate(...)` |

---

## 5. Мониторинг дрифта (Evidently)

Каждый вызов `/predict` записывает признаки + предсказание + вероятность в кольцевой буфер (до 10 000 записей).

Каждые 5 минут фоновая корутина `drift_monitoring_cron` строит отчёт `DataDriftPreset` (текущие данные vs. референсная выборка из обучающего датасета) и отправляет его в Evidently Workspace по адресу `http://158.160.2.37:8000/`.

Отчёт не строится, если накоплено менее 50 запросов.

**Project ID:** `019d3a16-ea3d-7be7-aff0-9e18e7b4e873`

Мониторируемые колонки: `race`, `sex`, `native.country`, `education`, `occupation`, `capital.gain`, `prediction`, `probability`.
