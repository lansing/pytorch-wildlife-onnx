.PHONY: install uninstall clean test lint export demo dataset-build dataset-clean \
        sweep-export sweep-eval

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

##
## Sweep targets
##

# Shared Docker invocation for GPU-dependent targets (eval / export / sweep).
# Mounts the full repo at /app so all internal paths resolve correctly.
IMAGE_TRT            ?= pytorch-wildlife-export-trt
CONTAINER_DATA_DIR   ?= /data/md_ft
CONTAINER_MODELS_DIR ?= /exported_models
CONTAINER_CACHE_DIR  ?= /root/.cache/pytorch_wildlife_export

DOCKER_RUN = docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
	--network host \
	-v "$(CURDIR)/data:/data" \
	-v "$(CURDIR)/exported_models:$(CONTAINER_MODELS_DIR)" \
	-v "$(CURDIR)/cache:$(CONTAINER_CACHE_DIR)" \
	-v "$(CURDIR):/app" \
	--workdir /app \
	--entrypoint python3 \
	$(IMAGE_TRT)

## sweep-export — export all standard MDV6-yolov10 variants (float16+int8 × 640+320 × onnx+trt).
## Optional overrides:
##   SWEEP_MODELS="MDV6-yolov10-e"   SWEEP_SIZES="640"
##   SWEEP_FORMATS="float16"          SWEEP_RUNTIMES="tensorrt"
##   SWEEP_CALIB_IMAGES=200           SWEEP_DATASET_YAML=/data/md_ft/megadetector_ft.yaml
##   SWEEP_SKIP_EXISTING=--skip-existing   SWEEP_DRY_RUN=--dry-run
SWEEP_MODELS        ?=
SWEEP_SIZES         ?=
SWEEP_FORMATS       ?=
SWEEP_RUNTIMES      ?=
SWEEP_CALIB_IMAGES  ?= 100
SWEEP_DATASET_YAML  ?=
SWEEP_SKIP_EXISTING ?=
SWEEP_DRY_RUN       ?=

sweep-export:
	$(DOCKER_RUN) \
		-m PytorchWildlife_Export.sweep_export \
		--output-dir $(CONTAINER_MODELS_DIR) \
		--num-calib-images $(SWEEP_CALIB_IMAGES) \
		$(if $(SWEEP_MODELS),   --models   $(SWEEP_MODELS)) \
		$(if $(SWEEP_SIZES),    --sizes    $(SWEEP_SIZES)) \
		$(if $(SWEEP_FORMATS),  --formats  $(SWEEP_FORMATS)) \
		$(if $(SWEEP_RUNTIMES), --runtimes $(SWEEP_RUNTIMES)) \
		$(if $(SWEEP_DATASET_YAML), --dataset-yaml $(SWEEP_DATASET_YAML)) \
		$(SWEEP_SKIP_EXISTING) \
		$(SWEEP_DRY_RUN)

## sweep-eval — evaluate all present MDV6-yolov10 variants, print metrics, write CSV.
## Optional overrides:
##   SWEEP_EVAL_SPLIT=test
##   SWEEP_EVAL_MODELS="MDV6-yolov10-e"   SWEEP_EVAL_SIZES="640"
##   SWEEP_EVAL_FORMATS="float16 int8"     SWEEP_EVAL_RUNTIMES="tensorrt"
##   SWEEP_EVAL_OUT=/exported_models/my_results.csv
##   SWEEP_EVAL_LOG=INFO    SWEEP_EVAL_VERBOSE=--verbose
SWEEP_EVAL_SPLIT    ?= val
SWEEP_EVAL_MODELS   ?=
SWEEP_EVAL_SIZES    ?=
SWEEP_EVAL_FORMATS  ?=
SWEEP_EVAL_RUNTIMES ?=
SWEEP_EVAL_OUT      ?=
SWEEP_EVAL_LOG      ?= WARNING
SWEEP_EVAL_VERBOSE  ?=

sweep-eval:
	@echo "--- Sweep eval (split=$(SWEEP_EVAL_SPLIT)) ---"
	$(DOCKER_RUN) \
		-m PytorchWildlife_Export.sweep_eval \
		--models-dir $(CONTAINER_MODELS_DIR) \
		--dataset    $(CONTAINER_DATA_DIR)/megadetector_ft.yaml \
		--split      $(SWEEP_EVAL_SPLIT) \
		--log-level  $(SWEEP_EVAL_LOG) \
		$(if $(SWEEP_EVAL_MODELS),   --models   $(SWEEP_EVAL_MODELS)) \
		$(if $(SWEEP_EVAL_SIZES),    --sizes    $(SWEEP_EVAL_SIZES)) \
		$(if $(SWEEP_EVAL_FORMATS),  --formats  $(SWEEP_EVAL_FORMATS)) \
		$(if $(SWEEP_EVAL_RUNTIMES), --runtimes $(SWEEP_EVAL_RUNTIMES)) \
		$(if $(SWEEP_EVAL_OUT),      --out      $(SWEEP_EVAL_OUT)) \
		$(SWEEP_EVAL_VERBOSE)

# Default target
all: install test
