import uvicorn

from .app import create_app
from .config import load_config


def main():
    config = load_config()
    uvicorn.run(create_app(config), host="127.0.0.1", port=config.port, log_level="info")


if __name__ == "__main__":
    main()
