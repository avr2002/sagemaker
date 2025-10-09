"""ModelHandler defines a model handler for penguins classification using Keras/TensorFlow models"""

import io
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import joblib
import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn.compose import ColumnTransformer

REQUIRED_COLUMNS = [
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]


class ModelHandler(object):
    """A penguins classification model handler implementation."""

    def __init__(self):
        self.initialized = False
        self.model_dir = None
        self.features_transformer: Optional[ColumnTransformer] = None  # type: ignore[annotation-unchecked]
        self.target_classes: Optional[Iterable[ColumnTransformer]] = None  # type: ignore[annotation-unchecked]
        self.model: Optional[Sequential] = None  # type: ignore[annotation-unchecked]

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir")
        self.model_dir = Path(model_dir)

        logging.info(f"Loading model from: {self.model_dir}")

        try:
            # Load features transformer
            features_transformer_path = self.model_dir / "features_transformer.joblib"
            if features_transformer_path.exists():
                self.features_transformer = joblib.load(features_transformer_path)
                logging.info("Features transformer loaded successfully")
            else:
                raise FileNotFoundError(f"Features transformer not found at {features_transformer_path}")

            # Load target transformer to get classes
            target_transformer_path = self.model_dir / "target_transformer.joblib"
            if target_transformer_path.exists():
                target_transformer = joblib.load(target_transformer_path)
                try:
                    self.target_classes = target_transformer.named_transformers_["species"].categories_[0]
                except Exception:
                    self.target_classes = getattr(target_transformer, "categories_", [None])[0]
                if self.target_classes is None:
                    raise RuntimeError("Could not infer target classes from target_transformer")
                logging.info(f"Target classes loaded: {self.target_classes}")
            else:
                raise FileNotFoundError(f"Target transformer not found at {target_transformer_path}")

            # Load Keras model
            model_path = self.model_dir / "penguins.keras"
            if model_path.exists():
                self.model = keras.models.load_model(model_path, compile=False)
                logging.info(f"Model loaded successfully from {model_path}")
            else:
                files = list(self.model_dir.glob("**/*"))
                logging.error(f"Files in {self.model_dir}:\n\t{files}\n")
                raise FileNotFoundError(f"No penguins.keras model found in {self.model_dir}")

        except Exception as e:
            logging.error(f"Error during model initialization: {e}")
            raise

    def _validate_and_order_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and order DataFrame columns according to required schema."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df[REQUIRED_COLUMNS]

    def _parse_input_data(self, input_data) -> pd.DataFrame:
        """Parse input data from various formats into a DataFrame."""
        # Handle different input formats
        if isinstance(input_data, (bytes, bytearray)):
            input_str = input_data.decode("utf-8")
        elif isinstance(input_data, str):
            input_str = input_data
        else:
            # Assume it's already parsed JSON/dict
            input_str = json.dumps(input_data)

        # Parse JSON input
        try:
            payload = json.loads(input_str)

            # Handle different JSON structures
            if isinstance(payload, dict) and "instances" in payload:
                rows = payload["instances"]
            elif isinstance(payload, list):
                rows = payload
            else:
                rows = [payload]

            df = pd.DataFrame(rows)

        except json.JSONDecodeError:
            # Try parsing as CSV
            df = pd.read_csv(io.StringIO(input_str))
            # Handle case where CSV has unnamed columns but correct number
            if len(df.columns) == len(REQUIRED_COLUMNS) and any(str(c).startswith("Unnamed") for c in df.columns):
                df.columns = REQUIRED_COLUMNS

        return df

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        processed_data = []

        for data in request:
            try:
                # Extract the input data
                if isinstance(data, dict) and "body" in data:
                    input_data = data["body"]
                else:
                    input_data = data

                # Parse input data into DataFrame
                df = self._parse_input_data(input_data)

                # Validate and order columns
                ordered_df = self._validate_and_order_columns(df)

                # Transform using the loaded features transformer
                if self.features_transformer is None:
                    raise RuntimeError("Features transformer not loaded")

                X = self.features_transformer.transform(ordered_df)
                processed_data.append(np.asarray(X))

            except Exception as e:
                logging.error(f"Error preprocessing data: {e}")
                raise

        return processed_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        predictions = []

        for data in model_input:
            try:
                # Make prediction
                pred = self.model.predict(data)
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Error during inference: {e}")
                raise

        return predictions

    def postprocess(self, inference_output):
        """
        Return predict result in structured format.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        results = []

        for preds in inference_output:
            preds = np.asarray(preds)
            if preds.ndim == 1:
                preds = preds.reshape(1, -1)

            batch_results = []

            if preds.size == 0:
                batch_results = []
            elif preds.shape[1] > 1 and self.target_classes is not None:
                # Multi-class classification
                for row in preds:
                    idx = int(np.argmax(row))
                    prediction = str(self.target_classes[idx])
                    confidence = float(row[idx])
                    batch_results.append(
                        {
                            "prediction": prediction,
                            "confidence": confidence,
                            "probabilities": {
                                str(cls): float(prob) for cls, prob in zip(self.target_classes, row, strict=False)
                            },
                        }
                    )
            else:
                # Single output regression or binary classification
                for row in preds:
                    score = float(row[0]) if row.size else float("nan")
                    batch_results.append({"prediction": score})

            # If single prediction, return it directly, otherwise return list
            if len(batch_results) == 1:
                results.extend(batch_results)
            else:
                results.append(batch_results)

        return results

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: model server context
        """
        try:
            model_input = self.preprocess(data)
            model_out = self.inference(model_input)
            return self.postprocess(model_out)
        except Exception as e:
            logging.error(f"Error in handle method: {e}")
            # Return error in same format as success response
            return [{"error": str(e)}]


# Global service instance
_service = ModelHandler()


def handle(data, context):
    """Entry point for the model server"""
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
