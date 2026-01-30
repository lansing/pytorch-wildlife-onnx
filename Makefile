.PHONY: install uninstall clean test lint export demo

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
	$(VENV_DIR)/bin/uv pip install -r $(PWE_DIR)/requirements.txt
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

# Default target
all: install test
