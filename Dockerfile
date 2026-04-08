FROM python:3.10-bullseye

WORKDIR /app

# Install dependencies as Root
COPY pyproject.toml .
RUN pip install uv && \
    uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    uv pip install --system openenv-core pydantic numpy opencv-python-headless python-dotenv requests "openai>=1.0.0" "transformers>=4.40.0" Pillow

# Setup Non-Root User for Hugging Face Spaces Security Policy
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
EXPOSE 7860

# Run the persistent web dashboard (keeps Space alive)
CMD ["python", "app.py"]
