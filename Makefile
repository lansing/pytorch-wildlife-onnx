.PHONY: install uninstall clean test lint export demo dataset-build dataset-clean

PYTHON_VERSION = 3.11.8
VENV_DIR = .venv
PWE_DIR = PytorchWildlife_Export

# Ensure pyenv is initialized in the shell
# If pyenv is not initialized, you might need to run:
# eval "$(pyenv init --path)"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"

install:
	@echo "--- Installing Python $(PYTHON_VERSION) with pyenv ---"
	pyenv install $(PYTHON_VERSION) --skip-existing
	pyenv local $(PYTHON_VERSION)
	@echo "--- Installing uv ---"
	pip install uv
	@echo "--- Creating and activating virtual environment with uv ---"
	uv venv
	@echo "--- Installing Python dependencies ---"
	uv pip install -r $(PWE_DIR)/requirements.txt
	@echo "--- Setup complete. Activate with: source $(VENV_DIR)/bin/activate ---"

uninstall:
	@echo "--- Removing virtual environment ---"
	rm -rf $(VENV_DIR)
	@echo "--- Removing local pyenv Python version ---"
	# pyenv uninstall $(PYTHON_VERSION) # Uncomment if you want to remove the pyenv Python version as well
	@echo "--- Uninstall complete ---"

clean:
	@echo "--- Cleaning up generated files ---"
	rm -rf exported_models_test
	rm -rf $(PWE_DIR)/demo/demo_output
	rm -rf $(PWE_DIR)/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.onnx" -delete # Remove exported ONNX models
	@echo "--- Clean up complete ---"

test:
	@echo "--- Running all unit tests ---"
	source $(VENV_DIR)/bin/activate && python -m unittest discover $(PWE_DIR)/tests

lint:
	@echo "--- Running linter (placeholder) ---"
	# uv run pylint $(PWE_DIR) # Uncomment and install pylint if desired
	@echo "--- Linting complete ---"

export:
	@echo "--- Exporting sample YOLOv9 model (float32, simplified, 1280x1280, raw output) ---"
	source $(VENV_DIR)/bin/activate && python $(PWE_DIR)/export_tool.py \
		--model_type yolov9 \
		--model_version MDV6-yolov9-c \
		--output_path exported_models_test/MDV6-yolov9-c_1280x1280_raw.onnx \
		--format float32 \
		--opset 18 \
		--simplify \
		--input_img_size 1280

demo:
	@echo "--- Running ONNX image detector demo ---"
	source $(VENV_DIR)/bin/activate && python $(PWE_DIR)/demo/onnx_image_detector_demo.py

docs:
	@echo "Documentation can be found in $(PWE_DIR)/docs/model_postprocessing.md"
	@echo "You can open it with a Markdown viewer or editor."

##
## Dataset targets
##

# Wipe existing WCS / CCT data and rebuild from scratch with all 5 wildlife
# sources + COCO.  Targets ~5,000 images with location-aware 80/10/10 splits.
# Override any budget with e.g. make dataset-build WCS_MAX_ANIMAL=1200
WCS_MAX_ANIMAL    ?= 800
WCS_MAX_VEHICLE   ?= 50
WCS_MAX_EMPTY     ?= 200
CCT_MAX_ANIMAL    ?= 700
CCT_MAX_EMPTY     ?= 150
SS_MAX_ANIMAL     ?= 700
SS_MAX_EMPTY      ?= 150
ICT_MAX_ANIMAL    ?= 400
ICT_MAX_EMPTY     ?= 100
COCO_MAX_PERSON   ?= 400
COCO_MAX_VEHICLE  ?= 200
DATASET_OUTPUT    ?= data/md_ft
DATASET_WORKERS   ?= 8

DOCKER_IMAGE      ?= pytorch-wildlife-export-trt

# Cache dirs are stored on the host so annotation ZIPs survive container restarts.
WCS_CACHE_HOST    ?= $(HOME)/.cache/pytorch_wildlife_export/wcs
CCT_CACHE_HOST    ?= $(HOME)/.cache/pytorch_wildlife_export/cct
SS_CACHE_HOST     ?= $(HOME)/.cache/pytorch_wildlife_export/serengeti
ICT_CACHE_HOST    ?= $(HOME)/.cache/pytorch_wildlife_export/island_conservation
FO_CACHE_HOST     ?= $(HOME)/.fiftyone

dataset-clean:
	@echo "--- Removing existing WCS / CCT / md_ft data (via Docker to handle root-owned files) ---"
	@mkdir -p data
	docker run --rm \
		-v $(PWD)/data:/app/data \
		--entrypoint sh \
		$(DOCKER_IMAGE) -c "rm -rf /app/data/md_ft /app/data/cct_ood"
	@echo "--- Done ---"

dataset-build: dataset-clean
	@echo "--- Building MegaDetector fine-tuning dataset (inside Docker) ---"
	@mkdir -p $(WCS_CACHE_HOST) $(CCT_CACHE_HOST) $(SS_CACHE_HOST) $(ICT_CACHE_HOST) $(FO_CACHE_HOST) data
	docker run --rm \
		-v $(PWD)/PytorchWildlife_Export:/app/PytorchWildlife_Export:ro \
		-v $(PWD)/data:/app/data \
		-v $(WCS_CACHE_HOST):/root/.cache/pytorch_wildlife_export/wcs \
		-v $(CCT_CACHE_HOST):/root/.cache/pytorch_wildlife_export/cct \
		-v $(SS_CACHE_HOST):/root/.cache/pytorch_wildlife_export/serengeti \
		-v $(ICT_CACHE_HOST):/root/.cache/pytorch_wildlife_export/island_conservation \
		-v $(FO_CACHE_HOST):/root/.fiftyone \
		-w /app \
		--entrypoint python3 \
		$(DOCKER_IMAGE) \
		-m PytorchWildlife_Export.dataset.dataset_builder \
		--output-dir $(DATASET_OUTPUT) \
		--wcs-max-animal   $(WCS_MAX_ANIMAL) \
		--wcs-max-vehicle  $(WCS_MAX_VEHICLE) \
		--wcs-max-empty    $(WCS_MAX_EMPTY) \
		--cct-max-animal   $(CCT_MAX_ANIMAL) \
		--cct-max-empty    $(CCT_MAX_EMPTY) \
		--serengeti-max-animal $(SS_MAX_ANIMAL) \
		--serengeti-max-empty  $(SS_MAX_EMPTY) \
		--island-max-animal    $(ICT_MAX_ANIMAL) \
		--island-max-empty     $(ICT_MAX_EMPTY) \
		--coco-max-person      $(COCO_MAX_PERSON) \
		--coco-max-vehicle     $(COCO_MAX_VEHICLE) \
		--workers              $(DATASET_WORKERS) \
		--log-level INFO
	@echo "--- Dataset build complete.  Config: $(DATASET_OUTPUT)/megadetector_ft.yaml ---"
	@echo "--- Provenance: $(DATASET_OUTPUT)/data_readme.md ---"

# Default target
all: install test
