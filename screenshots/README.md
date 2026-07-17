# Streamlit App Screenshots

This directory provides visual documentation of the deployed Streamlit app for the main project README and gives maintainers a consistent plan for recapturing images after significant interface changes.

| PNG filename | App page or section | Required visible content |
| --- | --- | --- |
| `01_overview.png` | EnergyTypeNet — Overview | The expanded sidebar with building-feature sliders, all four top accuracy metric cards, and the test-accuracy bar chart in a full-width browser window. |
| `02_model_comparison.png` | EnergyTypeNet — Model Comparison | The complete 5-fold cross-validation table with CV mean, CV standard deviation and test accuracy for every model. |
| `03_decision_boundaries.png` | EnergyTypeNet — Decision Boundaries | The multi-panel PCA decision-boundary visualization, with PC1/PC2 axes, colored building-type regions and observed points overlaid. |
| `04_roc_curves.png` | EnergyTypeNet — ROC / AUC | ROC curves for all three building-type classes across the models, including clearly visible AUC annotations and legends. |
| `05_custom_dataset_analysis.png` | Custom Dataset after uploading a CSV | Use a clean non-sensitive classification CSV and show the dataset summary, selected target/features and one useful diagnostic view such as multiclass ROC curves. |
| `06_chat_assistant.png` | Custom Dataset — Dataset Assistant | At least three visible exchanges covering model performance and a follow-up question, the assistant responses, and suggested follow-up question chips. |
| `07_live_prediction.png` | EnergyTypeNet — Live Prediction | Informative sidebar slider values and the main prediction cards showing model outputs and probabilities for every building type. |

## How to Capture Screenshots

1. Open the deployed app at `https://energytypenet-ml.streamlit.app/`.
2. Navigate to each page and set inputs to values that make the screenshot informative.
3. Set the browser window to 1440 × 900 pixels or 1280 × 800 pixels for consistent sizing.
4. Use the browser's built-in screenshot function or the operating system screenshot tool.
5. Save each image as PNG using the exact filename from the table.
6. Place the PNG files in this `screenshots/` directory, replacing the existing images when recapturing them.
7. Preview the main `README.md` on GitHub and verify that every image renders correctly.

## When to Update Screenshots

Recapture the screenshots whenever the dashboard layout changes significantly, new models are added to the comparison page, or the chat assistant interface is updated.
