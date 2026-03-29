import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow.exceptions
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.metrics import (
    CURRENT_MODEL_INFO,
    FEATURE_VALUES,
    HTTP_ERRORS_TOTAL,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    MODEL_INFERENCE_DURATION,
    MODEL_PREDICTIONS_TOTAL,
    MODEL_PROBABILITY,
    MODEL_UPDATES_TOTAL,
    metrics_app,
    update_model_info,
)
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.monitoring import drift_monitoring_cron, init_monitoring, record_request
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)

logger = logging.getLogger(__name__)

MODEL = Model()

EVIDENTLY_PROJECT_ID = '019d3a16-ea3d-7be7-aff0-9e18e7b4e873'


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    update_model_info(run_id, list(MODEL.features))

    init_monitoring(EVIDENTLY_PROJECT_ID)
    asyncio.ensure_future(drift_monitoring_cron())

    yield


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    app.mount('/metrics', metrics_app)

    @app.middleware('http')
    async def metrics_middleware(request: Request, call_next):
        endpoint = request.url.path
        method = request.method
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start
        status = str(response.status_code)

        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=status).inc()
        HTTP_REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)

        if response.status_code >= 500:
            HTTP_ERRORS_TOTAL.labels(endpoint=endpoint).inc()

        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception('Unhandled exception on %s', request.url.path)
        HTTP_ERRORS_TOTAL.labels(endpoint=request.url.path).inc()
        return JSONResponse(status_code=500, content={'detail': 'Internal server error'})

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        return {'status': 'ok', 'run_id': model_state.run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model_state = MODEL.get()
        if model_state.model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        # Build dataframe for the model
        preprocess_start = time.perf_counter()
        try:
            df = to_dataframe(request, needed_columns=MODEL.features)
        except Exception as e:
            logger.warning('Preprocessing error: %s', e)
            raise HTTPException(status_code=422, detail=f'Feature preprocessing failed: {e}')

        missing = [col for col in MODEL.features if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f'Missing required features: {missing}',
            )

        preprocess_duration = time.perf_counter() - preprocess_start
        from ml_service.metrics import PREPROCESSING_DURATION
        PREPROCESSING_DURATION.observe(preprocess_duration)

        # Log numeric feature values
        for col in df.select_dtypes(include='number').columns:
            val = df[col].iloc[0]
            if val is not None:
                FEATURE_VALUES.labels(feature=col).observe(float(val))

        # Inference
        inference_start = time.perf_counter()
        try:
            probability = float(model_state.model.predict_proba(df)[0][1])
        except Exception as e:
            logger.error('Model inference error: %s', e)
            raise HTTPException(status_code=500, detail=f'Model inference failed: {e}')
        inference_duration = time.perf_counter() - inference_start

        prediction = int(probability >= 0.5)

        MODEL_INFERENCE_DURATION.observe(inference_duration)
        MODEL_PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()
        MODEL_PROBABILITY.observe(probability)

        # Accumulate for Evidently
        features_dict = df.iloc[0].to_dict()
        record_request(features_dict, prediction, probability)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id.strip()
        if not run_id:
            raise HTTPException(status_code=422, detail='run_id must not be empty')

        try:
            MODEL.set(run_id=run_id)
        except mlflow.exceptions.MlflowException as e:
            MODEL_UPDATES_TOTAL.labels(status='failure').inc()
            logger.warning('Model update failed for run_id=%s: %s', run_id, e)
            raise HTTPException(
                status_code=404,
                detail=f'Could not load model for run_id "{run_id}": {e}',
            )
        except Exception as e:
            MODEL_UPDATES_TOTAL.labels(status='failure').inc()
            logger.error('Unexpected error during model update: %s', e)
            raise HTTPException(status_code=500, detail=f'Model update failed: {e}')

        MODEL_UPDATES_TOTAL.labels(status='success').inc()
        update_model_info(run_id, list(MODEL.features))
        logger.info('Model updated to run_id=%s', run_id)

        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
