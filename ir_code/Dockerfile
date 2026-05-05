FROM kingsleycl/cityflow_env:latest

WORKDIR /workspace

# Keep runtime deterministic and non-interactive.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Optional helpers for metrics inspection.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir pandas seaborn

CMD ["bash"]
