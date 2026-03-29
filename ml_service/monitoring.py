"""
Evidently drift monitoring: accumulates requests and periodically
builds a DataDrift report, saving it to the remote Evidently workspace.
"""
import asyncio
import logging
from collections import deque
from threading import Lock

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

EVIDENTLY_URL = 'http://158.160.2.37:8000/'
EVIDENTLY_PROJECT_ID = None  # set via init_monitoring()

REPORT_INTERVAL_SECONDS = 300  # build a report every 5 minutes
MIN_SAMPLES_FOR_REPORT = 50    # don't build if fewer samples accumulated

_lock = Lock()
_buffer: deque[dict] = deque(maxlen=10_000)
_reference_data: pd.DataFrame | None = None

NUMERIC_FEATURES = ['capital.gain']
CAT_FEATURES = ['race', 'sex', 'native.country', 'education', 'occupation']
ALL_MONITORED_FEATURES = CAT_FEATURES + NUMERIC_FEATURES + ['prediction', 'probability']


def _load_reference_data() -> pd.DataFrame:
    """Build reference dataset from the original Adult Census Income training split."""
    dataset = load_dataset('scikit-learn/adult-census-income')
    df = dataset['train'].to_pandas()
    sample = df[CAT_FEATURES + NUMERIC_FEATURES].dropna().sample(
        n=min(1000, len(df)), random_state=42
    )
    sample['prediction'] = 0
    sample['probability'] = 0.0
    return sample


def init_monitoring(project_id: str) -> None:
    global EVIDENTLY_PROJECT_ID, _reference_data
    EVIDENTLY_PROJECT_ID = project_id
    try:
        _reference_data = _load_reference_data()
        logger.info('Evidently reference data loaded (%d rows)', len(_reference_data))
    except Exception as e:
        logger.warning('Could not load reference data: %s', e)


def record_request(features: dict, prediction: int, probability: float) -> None:
    """Called after each /predict to accumulate data for drift monitoring."""
    row = {col: features.get(col) for col in CAT_FEATURES + NUMERIC_FEATURES}
    row['prediction'] = prediction
    row['probability'] = probability
    with _lock:
        _buffer.append(row)


def _build_and_send_report() -> None:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import RemoteWorkspace

    with _lock:
        if len(_buffer) < MIN_SAMPLES_FOR_REPORT:
            logger.info('Not enough samples for drift report (%d)', len(_buffer))
            return
        current_data = pd.DataFrame(list(_buffer))
        _buffer.clear()

    if _reference_data is None:
        logger.warning('Reference data not loaded, skipping drift report')
        return

    ref = _reference_data[current_data.columns.intersection(_reference_data.columns)]
    cur = current_data[current_data.columns.intersection(_reference_data.columns)]

    try:
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=ref, current_data=cur)
        workspace = RemoteWorkspace(EVIDENTLY_URL)
        workspace.add_run(EVIDENTLY_PROJECT_ID, result)
        logger.info('Drift report sent to Evidently workspace')
    except Exception as e:
        logger.error('Failed to send drift report: %s', e)


async def drift_monitoring_cron() -> None:
    """Background coroutine: builds and sends a drift report every REPORT_INTERVAL_SECONDS."""
    logger.info(
        'Drift monitoring cron started (interval=%ds)', REPORT_INTERVAL_SECONDS
    )
    while True:
        await asyncio.sleep(REPORT_INTERVAL_SECONDS)
        try:
            _build_and_send_report()
        except Exception as e:
            logger.error('Drift cron error: %s', e)
