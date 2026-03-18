.PHONY: run-all download-dataset generate-features train-model run-model-on-test-data

run-all: download-dataset generate-features train-model run-model-on-test-data

download-dataset:
	uv sync
	mkdir -p dataset
	uv run kaggle competitions download -c favorita-grocery-sales-forecasting --path dataset
	unzip -n dataset/favorita-grocery-sales-forecasting.zip -d dataset
	find "dataset" -type f -name "*.7z" | xargs -n1 -I % 7z x % -odataset -aos

generate-features:
	uv run python src/prepare_features.py

train-model:
	uv run python src/train_model.py --mode=eval

run-model-on-test-data:
	uv run python src/train_model.py --mode=test
