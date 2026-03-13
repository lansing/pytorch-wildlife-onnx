.PHONY: install uninstall clean test lint export demo \
        dataset-download-wcs-test dataset-download-coco-test dataset-build-test \
        dataset-build eval-baseline eval sweep-export sweep-eval

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

# ---------------------------------------------------------------------------
# Dataset targets
# ---------------------------------------------------------------------------
# All dataset targets run inside the pytorch-wildlife-export-trt Docker image so
# they share the same Python environment as the export / eval pipeline.
#
# Directory layout (on host, mounted read-write into the container):
#   data/md_ft/       ← assembled YOLO dataset (images/, labels/, YAML)
#   cache/wcs/        ← WCS annotation JSON cache (~80 MB, reused across runs)
#   exported_models/  ← TRT engine files for eval
#
# Symlink note: the dataset builder uses symlinks from images/{split}/ into
# _raw/.  These paths use the container-internal prefix /data/md_ft, so the
# data/ directory must always be mounted at /data inside the container.

IMAGE_TRT            = pytorch-wildlife-export-trt
CONTAINER_DATA_DIR   = /data/md_ft
CONTAINER_MODELS_DIR = /exported_models
CONTAINER_CACHE_DIR  = /root/.cache/pytorch_wildlife_export
BASELINE_ENGINE      = MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.engine

# Base docker invocation shared by all dataset / eval targets
DOCKER_RUN = docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
	--network host \
	-v "$(CURDIR)/data:/data" \
	-v "$(CURDIR)/exported_models:$(CONTAINER_MODELS_DIR)" \
	-v "$(CURDIR)/cache:$(CONTAINER_CACHE_DIR)" \
	-v "$(CURDIR):/app" \
	--workdir /app \
	--entrypoint python3 \
	$(IMAGE_TRT)

## dataset-download-wcs-test
##   Download a small WCS subset: 100 animal-primary + 100 vehicle-primary images.
##   Uses only stdlib HTTP — no extra packages needed.  ~150–300 MB download.
dataset-download-wcs-test:
	@mkdir -p data/md_ft cache/wcs
	@echo "--- Downloading WCS test subset (100 animal + 100 vehicle images) ---"
	$(DOCKER_RUN) \
		-m PytorchWildlife_Export.dataset.dataset_builder \
		--output-dir $(CONTAINER_DATA_DIR) \
		--wcs-max-animal 100 \
		--wcs-max-vehicle 100 \
		--skip-coco \
		--log-level INFO

## dataset-download-coco-test
##   Download a small COCO 2017 subset: 100 person + 100 vehicle images.
##   Requires fiftyone — installed on-the-fly inside the container (~500 MB, slow on first run).
##   fiftyone is cached inside the container layer only; re-run will reinstall unless
##   you add fiftyone to the Docker image.
dataset-download-coco-test:
	@mkdir -p data/md_ft
	@echo "--- Downloading COCO test subset (100 person + 100 vehicle images) ---"
	@echo "    Installing fiftyone inside the container (first run is slow) ..."
	docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
		--network host \
		-v "$(CURDIR)/data:/data" \
		-v "$(CURDIR)/cache:$(CONTAINER_CACHE_DIR)" \
		-v "$(CURDIR):/app" \
		--workdir /app \
		--entrypoint bash \
		$(IMAGE_TRT) \
		-c "pip install fiftyone -q && python3 \
		    -m PytorchWildlife_Export.dataset.dataset_builder \
		    --output-dir $(CONTAINER_DATA_DIR) \
		    --coco-max-person 100 \
		    --coco-max-vehicle 100 \
		    --skip-wcs \
		    --log-level INFO"

## dataset-build-test
##   Full test build: WCS (100 animal + 100 vehicle) + COCO (100 person + 100 vehicle).
##   Total download ~300–500 MB.  Installs fiftyone on-the-fly for the COCO portion.
dataset-build-test:
	@mkdir -p data/md_ft cache/wcs
	@echo "--- Building test dataset (100 images/class, WCS + COCO) ---"
	@echo "    Installing fiftyone inside the container (first run is slow) ..."
	docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
		--network host \
		-v "$(CURDIR)/data:/data" \
		-v "$(CURDIR)/exported_models:$(CONTAINER_MODELS_DIR)" \
		-v "$(CURDIR)/cache:$(CONTAINER_CACHE_DIR)" \
		-v "$(CURDIR):/app" \
		--workdir /app \
		--entrypoint bash \
		$(IMAGE_TRT) \
		-c "pip install fiftyone -q && python3 \
		    -m PytorchWildlife_Export.dataset.dataset_builder \
		    --output-dir $(CONTAINER_DATA_DIR) \
		    --wcs-max-animal 100 \
		    --wcs-max-vehicle 100 \
		    --coco-max-person 100 \
		    --coco-max-vehicle 100 \
		    --log-level INFO"

## dataset-build
##   Full production dataset: WCS (2500 animal + 500 vehicle) + COCO (1500 person + 500 vehicle).
##   BUDGET WARNING: ~4–5 GB download.  Do not run without verifying available disk space.
##   Installs fiftyone on-the-fly for the COCO portion.
dataset-build:
	@mkdir -p data/md_ft cache/wcs
	@echo "--- Building full fine-tuning dataset ---"
	@echo "    BUDGET: ~4–5 GB expected.  Press Ctrl-C within 5 s to abort."
	@sleep 5
	docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
		--network host \
		-v "$(CURDIR)/data:/data" \
		-v "$(CURDIR)/exported_models:$(CONTAINER_MODELS_DIR)" \
		-v "$(CURDIR)/cache:$(CONTAINER_CACHE_DIR)" \
		-v "$(CURDIR):/app" \
		--workdir /app \
		--entrypoint bash \
		$(IMAGE_TRT) \
		-c "pip install fiftyone -q && python3 \
		    -m PytorchWildlife_Export.dataset.dataset_builder \
		    --output-dir $(CONTAINER_DATA_DIR) \
		    --wcs-max-animal 2500 \
		    --wcs-max-vehicle 500 \
		    --coco-max-person 1500 \
		    --coco-max-vehicle 500 \
		    --log-level INFO"

## eval-baseline
##   Evaluate the float16 TRT baseline engine against the val split.
##   Requires the dataset to have been built first (make dataset-build-test or dataset-build).
eval-baseline:
	@echo "--- Evaluating baseline: $(BASELINE_ENGINE) ---"
	$(DOCKER_RUN) \
		-m PytorchWildlife_Export.dataset.eval \
		$(CONTAINER_MODELS_DIR)/$(BASELINE_ENGINE) \
		--dataset $(CONTAINER_DATA_DIR)/megadetector_ft.yaml \
		--split val \
		--log-level INFO

## eval MODEL=<filename>
##   Evaluate any model in exported_models/ against the val split.
##   MODEL can be a .engine or .onnx file (filename only, no path).
##   Optional: SPLIT=test  CONF=0.1
##
##   Examples:
##     make eval MODEL=MDV6-yolov10-e_int8_640_denorm_nhwc_uint8input.engine
##     make eval MODEL=MDV6-yolov10-e_float32_1280_raw.onnx SPLIT=test
MODEL ?= $(BASELINE_ENGINE)
SPLIT ?= val
CONF  ?= 0.1
eval:
	@echo "--- Evaluating: $(MODEL) (split=$(SPLIT), conf=$(CONF)) ---"
	$(DOCKER_RUN) \
		-m PytorchWildlife_Export.dataset.eval \
		$(CONTAINER_MODELS_DIR)/$(MODEL) \
		--dataset $(CONTAINER_DATA_DIR)/megadetector_ft.yaml \
		--split $(SPLIT) \
		--conf $(CONF) \
		--log-level INFO

## sweep-export
##   Export all standard MDV6-yolov10 variants (e/c × 640/320 × float16/int8 × onnx/trt).
##   All exports use uint8+nhwc+denorm preprocessing.  INT8 ONNX is skipped (no benefit
##   without TRT fusion).  Total: 12 artifacts (4 float16-onnx, 4 float16-trt, 4 int8-trt).
##
##   Optional overrides (space-separated, quote if needed):
##     SWEEP_MODELS="MDV6-yolov10-e"
##     SWEEP_SIZES="640"
##     SWEEP_FORMATS="float16"
##     SWEEP_RUNTIMES="tensorrt"
##     SWEEP_CALIB_IMAGES=200
##     SWEEP_SKIP_EXISTING=--skip-existing
##     SWEEP_DRY_RUN=--dry-run
##
##   Example — dry-run to preview filenames:
##     make sweep-export SWEEP_DRY_RUN=--dry-run
##
##   Example — rebuild only missing TRT engines:
##     make sweep-export SWEEP_RUNTIMES=tensorrt SWEEP_SKIP_EXISTING=--skip-existing
SWEEP_MODELS         ?=
SWEEP_SIZES          ?=
SWEEP_FORMATS        ?=
SWEEP_RUNTIMES       ?=
SWEEP_CALIB_IMAGES   ?= 100
SWEEP_DATASET_YAML   ?=
SWEEP_SKIP_EXISTING  ?=
SWEEP_DRY_RUN        ?=

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

## sweep-eval
##   Evaluate all standard MDV6-yolov10 variants present in exported_models/.
##   Missing files are skipped.  Prints metrics as each model completes, then
##   writes a CSV summary to exported_models/eval_results_<split>.csv.
##
##   Optional overrides:
##     SWEEP_EVAL_SPLIT=test
##     SWEEP_EVAL_MODELS="MDV6-yolov10-e"
##     SWEEP_EVAL_SIZES="640"
##     SWEEP_EVAL_FORMATS="float16 int8"
##     SWEEP_EVAL_RUNTIMES="tensorrt"
##     SWEEP_EVAL_OUT=/exported_models/my_results.csv
##     SWEEP_EVAL_LOG=INFO        (default WARNING — suppress per-image noise)
##     SWEEP_EVAL_VERBOSE=--verbose  (print full per-model eval table)
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
