.PHONY: setup run test lint db-setup db-migrate docker-build docker-up docker-down clean

# Development setup
setup:
	pip install -r requirements.txt

# Run the API
run:
	uvicorn app.main:app --reload

# Run tests
test:
	pytest

# Run linting
lint:
	flake8 app tests

# Database setup
db-setup:
	alembic init alembic
	alembic revision --autogenerate -m "Initial migration"
	alembic upgrade head

# Create new migration
db-migrate:
	alembic revision --autogenerate -m "Migration $(shell date +%Y%m%d%H%M%S)"
	alembic upgrade head

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Clean temporary files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Help
help:
	@echo "make setup      - Install dependencies"
	@echo "make run        - Run the API"
	@echo "make test       - Run tests"
	@echo "make lint       - Run linting"
	@echo "make db-setup   - Initialize database and run migrations"
	@echo "make db-migrate - Create a new migration"
	@echo "make docker-build - Build Docker images"
	@echo "make docker-up  - Start all Docker containers"
	@echo "make docker-down - Stop all Docker containers"
	@echo "make clean      - Remove temporary files"
