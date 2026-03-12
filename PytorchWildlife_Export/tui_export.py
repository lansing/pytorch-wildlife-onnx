import argparse  # Added argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Markdown,
    RadioButton,
    RadioSet,
    Static,
)

try:
    CONFIG = yaml.safe_load(Path("PytorchWildlife_Export/tui_config.yaml").read_text())
except FileNotFoundError:
    print(
        "Error: tui_config.yaml not found. Make sure it's in the PytorchWildlife_Export directory."
    )
    sys.exit(1)


class QuitScreen(ModalScreen):
    """Screen with a dialog to quit."""

    def compose(self) -> ComposeResult:
        yield Container(
            Container(
                Label("Are you sure you want to quit?", id="question"),
                Container(
                    Button("Quit", variant="error", id="quit"),
                    Button("Cancel", variant="primary", id="cancel"),
                    id="buttons",
                ),
                id="dialog",
            ),
            id="quit_screen_container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()


class BaseSelectionScreen(Screen):
    """Base screen for showing a list of options."""

    def __init__(self, key: str, **kwargs):
        super().__init__(**kwargs)
        self.key = key
        self.heading = CONFIG["headings"].get(key, "")
        self.instruction = CONFIG["instructions"].get(key, "")

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.heading, classes="heading"),
            Markdown(self.instruction),
            id="selection_container",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Container).border_title = self.heading

    def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.app.push_screen(QuitScreen())


class ChoiceSelectionScreen(BaseSelectionScreen):
    """Screen for selecting from a list of choices."""

    def __init__(self, key: str, next_screen_callable, **kwargs):
        super().__init__(key, **kwargs)
        self.next_screen_callable = next_screen_callable
        self._id_to_value = {}

    def compose(self) -> ComposeResult:
        # yield Header()
        with Container(id="selection_container"):
            # yield Label(self.heading, classes="heading")
            yield Markdown(self.instruction)
            options = self.get_options()
            with RadioSet(id=f"{self.key}_radioset"):
                for option in options:
                    option_id = str(option["value"])
                    self._id_to_value[option_id] = option["value"]
                    radio_button = RadioButton(option["label"], id=option_id)
                    if self.app.selections.get(self.key) == option["value"]:
                        radio_button.value = True
                    yield radio_button
            yield Button("Next", variant="primary", id="next_button")
        yield Footer()

    def get_options(self):
        if self.key == "model_version":
            model_type = self.app.selections.get("model_type")
            return CONFIG["options"]["model_version"].get(model_type, [])
        return CONFIG["options"].get(self.key, [])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next_button":
            radioset = self.query_one(f"#{self.key}_radioset", RadioSet)
            if radioset.pressed_button:
                selected_id = radioset.pressed_button.id
                self.app.selections[self.key] = self._id_to_value[selected_id]
                self.app.push_screen(self.next_screen_callable())

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed:
            selected_id = event.pressed.id
            self.app.selections[self.key] = self._id_to_value[selected_id]


class InputScreen(BaseSelectionScreen):
    """Screen for text or number input."""

    def __init__(self, key: str, next_screen_callable, **kwargs):
        super().__init__(key, **kwargs)
        self.next_screen_callable = next_screen_callable
        self.input_type = "number" if key in ["input_img_size", "num_calibration_images"] else "text"
        self.min_val = (
            CONFIG["ranges"][key].get("min") if self.input_type == "number" else None
        )
        self.max_val = (
            CONFIG["ranges"][key].get("max") if self.input_type == "number" else None
        )

    def compose(self) -> ComposeResult:
        with Container(id="selection_container"):
            yield Markdown(self.instruction)

            if self.key == "output_dir" and self.app.output_dir_cli is not None:
                yield Static(
                    f"Output directory set via CLI: [b]{self.app.output_dir_cli}[/b]"
                )
            else:
                yield Input(
                    value=str(self.app.selections.get(self.key, "")),
                    placeholder=f"e.g., {self.app.selections.get(self.key, '')}",
                    type=self.input_type,
                    id=f"{self.key}_input",
                )
                yield Label("", id="validation_label")
                yield Button("Next", variant="primary", id="next_button")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Container).border_title = self.heading  # Added back
        if self.key == "output_dir" and self.app.output_dir_cli is not None:
            self.app.selections[self.key] = self.app.output_dir_cli
            # Automatically advance to the next screen
            self.call_after_refresh(self.app.push_screen, self.next_screen_callable())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next_button":
            input_widget = self.query_one(f"#{self.key}_input", Input)
            value = input_widget.value
            if self.validate(value):
                self.app.selections[self.key] = (
                    int(value) if self.input_type == "number" else value
                )
                self.app.push_screen(self.next_screen_callable())
            else:
                self.query_one("#validation_label").update(
                    f"Please enter a valid value. Min: {self.min_val}, Max: {self.max_val}"
                )

    def validate(self, value) -> bool:
        if self.input_type == "number":
            try:
                num = int(value)
                return self.min_val <= num <= self.max_val
            except ValueError:
                return False
        return True


class SummaryScreen(Screen):
    """Screen to show a summary and run the export."""

    def compose(self) -> ComposeResult:
        with Container(id="summary_container"):
            yield Label(CONFIG["headings"]["run_export"], classes="heading")
            yield Markdown(CONFIG["instructions"]["run_export"])

            summary_text = self.get_summary_text()
            yield Markdown(summary_text, id="summary_markdown")

            yield Button("Run Export", variant="success", id="run_button")
            yield Button("Back", id="back_button")
        yield Footer()

    def get_summary_text(self) -> str:
        selections = self.app.selections
        output_path = self.get_output_path()

        preproc_parts = []
        if selections.get("denormalized_input"):
            preproc_parts.append("denormalized (0-255)")
        if selections.get("nhwc_input"):
            preproc_parts.append("NHWC layout")
        if selections.get("uint8_input"):
            preproc_parts.append("uint8 dtype")
        preproc_str = ", ".join(preproc_parts) if preproc_parts else "none"

        is_int8_trt = (
            selections.get("runtime") == "tensorrt"
            and selections.get("format") == "int8"
        )
        calib_line = (
            f"\n- **Calibration Images**: `{selections['num_calibration_images']}`"
            if is_int8_trt
            else ""
        )

        return f"""
- **Model Type**: `{selections["model_type"]}`
- **Model Version**: `{selections["model_version"]}`
- **Runtime**: `{selections["runtime"]}`
- **Output Directory**: `{selections["output_dir"]}`
- **Output Path**: `{output_path}`
- **Format**: `{selections["format"]}`
- **Input Image Size**: `{selections["input_img_size"]}`
- **Input Preprocessing**: `{preproc_str}`{calib_line}
"""

    def get_output_path(self) -> str:
        selections = self.app.selections
        filename_base = (
            f"{selections['model_version']}_{selections['format']}_"
            + f"{selections['input_img_size']}"
        )
        if selections["model_type"] == "yolov10_v9_compatible":
            filename_base += "_v9_compat"
        if selections.get("denormalized_input"):
            filename_base += "_denorm"
        if selections.get("nhwc_input"):
            filename_base += "_nhwc"
        if selections.get("uint8_input"):
            filename_base += "_uint8input"
        ext = ".engine" if selections.get("runtime") == "tensorrt" else ".onnx"
        filename = f"{filename_base}{ext}"
        return os.path.join(selections["output_dir"], filename)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run_button":
            self.app.selections["output_path"] = self.get_output_path()
            self.app.push_screen(ExecutionScreen())
        elif event.button.id == "back_button":
            self.app.pop_screen()


class ExecutionScreen(Screen):
    """Screen that runs the command and shows the output."""

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="log_container"):
            yield Static("Starting export...", id="log_header")
            yield Log(id="log_output", )
        yield Footer()

    async def on_mount(self) -> None:
        """Run the export command."""
        command = self.build_command()
        self.query_one("#log_header").update(" ".join(command))
        self.run_worker(self.execute_command(command), thread=False, exclusive=True)

    async def execute_command(self, command) -> None:
        log_output = self.query_one("#log_output", Log)
        log_output.write(
            "Model weights might need to be downloaded, which can take a while. Please hold...\n"
        )

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),  # Ensure unbuffered output
        )

        while True:
            line_bytes = await process.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace")
            log_output.write(line)

        await process.wait()

        if process.returncode == 0:
            self.query_one("#log_header").update("Export completed successfully!")
        else:
            self.query_one("#log_header").update(
                f"Export failed with exit code {process.returncode}."
            )

        self.query_one("VerticalScroll").scroll_end(animate=True)
        # Add a button to exit
        container = self.query_one("#log_container")
        await container.mount(Button("Exit", variant="primary", id="exit_button"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "exit_button":
            self.app.exit()

    def build_command(self) -> list[str]:
        selections = self.app.selections
        cmd = [
            "python",
            "PytorchWildlife_Export/export_tool.py",
            "--model_type",
            selections["model_type"],
            "--model_version",
            selections["model_version"],
            "--output_path",
            selections["output_path"],
            "--format",
            selections["format"],
            "--input_img_size",
            str(selections["input_img_size"]),
            "--opset",
            "18",
            "--simplify",
            "--runtime",
            selections["runtime"],
        ]
        if selections.get("denormalized_input"):
            cmd.append("--denormalized_input")
        if selections.get("nhwc_input"):
            cmd.append("--nhwc_input")
        if selections.get("uint8_input"):
            cmd.append("--uint8_input")
        if (
            selections.get("runtime") == "tensorrt"
            and selections.get("format") == "int8"
        ):
            cmd += [
                "--num_calibration_images",
                str(selections.get("num_calibration_images", 300)),
            ]
        return cmd


class ExportTUI(App):
    """A Textual application to walk through model export."""

    CSS_PATH = "tui_style.css"
    BINDINGS = [("q", "request_quit", "Quit")]

    def __init__(self, output_dir_cli: str = None, **kwargs):
        super().__init__(**kwargs)
        self.selections = self.load_defaults()
        self.output_dir_cli = output_dir_cli

    def load_defaults(self):
        defaults = CONFIG["defaults"].copy()
        # Set a default for model_version based on default model_type
        default_model_type = defaults["model_type"]
        defaults["model_version"] = CONFIG["options"]["model_version"][
            default_model_type
        ][0]["value"]
        return defaults

    def on_mount(self) -> None:
        """Start the wizard."""
        self.push_screen(self.get_model_type_screen())

    def get_model_type_screen(self):
        return ChoiceSelectionScreen("model_type", self.get_model_version_screen)

    def get_model_version_screen(self):
        return ChoiceSelectionScreen("model_version", self.get_runtime_screen)

    def get_runtime_screen(self):
        return ChoiceSelectionScreen("runtime", self.get_output_dir_screen)

    def get_output_dir_screen(self):
        return InputScreen("output_dir", self.get_format_screen)

    def get_format_screen(self):
        return ChoiceSelectionScreen("format", self.get_post_format_screen)

    def get_post_format_screen(self):
        if (
            self.selections.get("runtime") == "tensorrt"
            and self.selections.get("format") == "int8"
        ):
            return InputScreen("num_calibration_images", self.get_input_img_size_screen)
        return self.get_input_img_size_screen()

    def get_input_img_size_screen(self):
        return InputScreen("input_img_size", self.get_allow_denormalized_screen)

    def get_allow_denormalized_screen(self):
        return ChoiceSelectionScreen("denormalized_input", self.get_allow_nhwc_screen)

    def get_allow_nhwc_screen(self):
        return ChoiceSelectionScreen("nhwc_input", self.get_allow_uint8_screen)

    def get_allow_uint8_screen(self):
        return ChoiceSelectionScreen("uint8_input", self.get_summary_screen)

    def get_summary_screen(self):
        return SummaryScreen()

    def action_request_quit(self) -> None:
        """Action to display the quit dialog."""
        self.push_screen(QuitScreen())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Wildlife ONNX Export TUI")
    parser.add_argument(
        "--output-dir-cli",
        type=str,
        default=None,
        help="Specify an output directory from the CLI, bypassing interactive input.",
    )
    args = parser.parse_args()

    # Workaround for pythonw on macOS
    if sys.platform == "darwin" and "pythonw" in sys.executable:
        subprocess.run([sys.executable.replace("pythonw", "python"), *sys.argv])
        sys.exit(0)

    # Need to run from the root of the project
    os.chdir(Path(__file__).parent.parent)

    import asyncio

    app = ExportTUI(output_dir_cli=args.output_dir_cli)
    app.run()
