install:
	pip install -r requirements.txt

lint:
	flake8 src tests

format:
	black src tests

test:
	pytest tests

serve:
	python src/serve_model.py

docker-build:
	docker build -t stroke-predictor .

docker-run:
	docker run -p 8000:8000 stroke-predictor
