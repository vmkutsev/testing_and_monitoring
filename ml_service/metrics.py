"""
Prometheus metrics definitions for the ML service.
"""
from prometheus_client import Counter, Histogram, Info, make_asgi_app

# ── HTTP ──────────────────────────────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

HTTP_ERRORS_TOTAL = Counter(
    'http_errors_total',
    'Total number of HTTP 5XX errors',
    ['endpoint'],
)

# ── Preprocessing ─────────────────────────────────────────────────────────────

PREPROCESSING_DURATION = Histogram(
    'preprocessing_duration_seconds',
    'Time spent on input preprocessing',
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
)

FEATURE_VALUES = Histogram(
    'feature_value',
    'Distribution of numeric feature values passed to the model',
    ['feature'],
    buckets=[0, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
)

# ── Model inference ───────────────────────────────────────────────────────────

MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Time spent on model inference',
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

MODEL_PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total number of model predictions by class',
    ['prediction'],
)

MODEL_PROBABILITY = Histogram(
    'model_prediction_probability',
    'Distribution of predicted probabilities',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Model info ────────────────────────────────────────────────────────────────

MODEL_UPDATES_TOTAL = Counter(
    'model_updates_total',
    'Total number of model update operations',
    ['status'],  # success / failure
)

CURRENT_MODEL_INFO = Info(
    'current_model',
    'Information about the currently loaded model',
)


def update_model_info(run_id: str, features: list[str]) -> None:
    CURRENT_MODEL_INFO.info({
        'run_id': run_id,
        'features': ','.join(features),
        'num_features': str(len(features)),
    })


metrics_app = make_asgi_app()
