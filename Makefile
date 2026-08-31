.PHONY: all help install build run start dev clean test

# Default target
all: run

help:
	@echo "SonarVision Makefile Commands:"
	@echo "  make run        - Build frontend (if needed) and start unified app (port 8000)"
	@echo "  make dev        - Start backend and frontend dev servers concurrently"
	@echo "  make build      - Build frontend production bundle (frontend/dist)"
	@echo "  make install    - Install Python and Node.js dependencies"
	@echo "  make test       - Run backend test suite"
	@echo "  make clean      - Remove build artifacts and temporary cache files"

install:
	pip install -r requirements.txt
	cd frontend && npm install

build:
	cd frontend && npm install && npm run build

run start:
	./start.sh

dev:
	./start.sh --dev

test:
	pytest tests/

clean:
	rm -rf frontend/dist frontend/node_modules/.vite
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
