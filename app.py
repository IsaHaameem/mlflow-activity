"""
app.py
------
Gradio web interface for real-time inference using the exported XGBoost model (model.pkl).
Deployable locally and on cloud platforms such as Render and Hugging Face Spaces.
"""

import os
from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Global model variable loaded once on startup
_LOADED_MODEL = None
_LOAD_ERROR = None

try:
    if MODEL_PATH.exists():
        _LOADED_MODEL = joblib.load(MODEL_PATH)
    else:
        _LOAD_ERROR = (
            f"Model file '{MODEL_PATH.name}' was not found.\n"
            "Please run 'python export_model.py' to generate model.pkl before running predictions."
        )
except Exception as e:
    _LOAD_ERROR = f"Failed to load '{MODEL_PATH.name}': {e}"


def get_model():
    """
    Returns the loaded model or raises a descriptive RuntimeError.
    """
    global _LOADED_MODEL, _LOAD_ERROR
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL
    if _LOAD_ERROR:
        raise RuntimeError(_LOAD_ERROR)
    if MODEL_PATH.exists():
        _LOADED_MODEL = joblib.load(MODEL_PATH)
        return _LOADED_MODEL
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH.name}' was not found. "
        "Please run 'python export_model.py' to generate model.pkl."
    )


def predict(feature1, feature2, model=None):
    """
    Validates input features and returns predicted class and probabilities.

    Parameters:
        feature1: Numerical value for feature 1
        feature2: Numerical value for feature 2
        model: Optional model instance (defaults to preloaded model)

    Returns:
        tuple: (result_text, status_message)
    """
    # 1. Validate inputs
    if feature1 is None or feature2 is None:
        return (
            "Error: Both Feature 1 and Feature 2 are required.",
            "Validation Failed: Missing input values.",
        )

    try:
        f1 = float(feature1)
        f2 = float(feature2)
    except (ValueError, TypeError):
        return (
            "Error: Inputs must be valid numerical values.",
            "Validation Failed: Non-numeric input encountered.",
        )

    # 2. Acquire model
    try:
        active_model = model if model is not None else get_model()
    except Exception as err:
        return (
            f"Error: {err}",
            "Model Error: Unable to access trained model artifact.",
        )

    # 3. Create DataFrame with schema matching training data
    input_df = pd.DataFrame([{"feature1": f1, "feature2": f2}])

    # 4. Generate prediction
    try:
        pred_class = int(active_model.predict(input_df)[0])

        proba_text = ""
        if hasattr(active_model, "predict_proba"):
            probabilities = active_model.predict_proba(input_df)[0]
            confidence = probabilities[pred_class] * 100
            proba_text = f" (Confidence: {confidence:.1f}%)"

        result_summary = (
            f"Predicted Class: {pred_class}{proba_text}\n"
            f"Interpretation: Input point ({f1}, {f2}) is classified as Class {pred_class}."
        )
        status_info = "Prediction successful."
        return result_summary, status_info
    except Exception as err:
        return (
            f"Error during inference: {err}",
            "Inference Error: Model failed to process inputs.",
        )


def create_ui():
    """
    Constructs the Gradio interface.
    """
    title = "XGBoost Model Prediction"
    description = (
        "Enter two numerical features to obtain a predicted class from the "
        "trained machine learning model."
    )
    explanation = (
        "### About this Application\n"
        "- **Model**: Binary XGBoost Classifier trained on `feature1` and `feature2`.\n"
        "- **Classes**: `0` (lower range) or `1` (higher range).\n"
        "- **Lifecycle**: Experiment tracked in MLflow and exported to `model.pkl`.\n\n"
        "> **Note on Dataset**: This model was trained on a 10-sample demonstration dataset "
        "for MLflow tracking education. Predictions on real-world data should not be used "
        "in production without retraining on domain-specific data."
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        if _LOAD_ERROR:
            gr.Warning(_LOAD_ERROR)

        with gr.Row():
            with gr.Column(scale=1):
                f1_input = gr.Number(
                    label="Feature 1",
                    value=3.0,
                    info="Numerical value for feature 1 (e.g. 1 to 10)",
                )
                f2_input = gr.Number(
                    label="Feature 2",
                    value=13.0,
                    info="Numerical value for feature 2 (e.g. 11 to 20)",
                )
                submit_btn = gr.Button("Predict", variant="primary")

            with gr.Column(scale=1):
                output_box = gr.Textbox(
                    label="Prediction Output",
                    interactive=False,
                    lines=3,
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                )

        submit_btn.click(
            fn=lambda f1, f2: predict(f1, f2),
            inputs=[f1_input, f2_input],
            outputs=[output_box, status_box],
        )

        gr.Examples(
            examples=[
                [2.0, 12.0],
                [4.0, 14.0],
                [7.0, 17.0],
                [9.0, 19.0],
            ],
            inputs=[f1_input, f2_input],
            outputs=[output_box, status_box],
            fn=lambda f1, f2: predict(f1, f2),
            cache_examples=False,
        )

        gr.Markdown("---")
        gr.Markdown(explanation)

    return demo


# Create app instance for Gradio / Render / Hugging Face Spaces
app = create_ui()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(server_name="0.0.0.0", server_port=port)

