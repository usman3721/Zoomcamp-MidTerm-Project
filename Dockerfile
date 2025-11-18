FROM python:3.14-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code  
ENV PATH="/code/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --locked

COPY . .

EXPOSE 9696

ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "9696"]
