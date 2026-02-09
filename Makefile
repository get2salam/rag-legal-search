.PHONY: build run test lint stop clean

# Build the Docker image
build:
	docker compose build

# Run the full stack (app + ChromaDB)
run:
	docker compose up -d

# Run tests locally
test:
	pytest tests/ -v --tb=short

# Lint with ruff
lint:
	ruff check .
	ruff format --check .

# Stop all containers
stop:
	docker compose down

# Stop and remove volumes
clean:
	docker compose down -v

# View logs
logs:
	docker compose logs -f

# Rebuild and run
rebuild: stop build run
