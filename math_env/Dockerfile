FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "openenv-core[core]>=0.2.2"

EXPOSE 8000

CMD ["python", "-c", "from openenv.core.env_server.http_server import create_app; from server import NetZeroEnv, NetZeroAction, NetZeroObservation; import uvicorn; app=create_app(NetZeroEnv, NetZeroAction, NetZeroObservation, env_name='net_zero_nexus', max_concurrent_envs=1); uvicorn.run(app, host='0.0.0.0', port=8000)"]
