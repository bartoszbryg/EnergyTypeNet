# Energy Consumption Dataset
Description: The Energy Consumption Dataset provides information on energy consumption across different building types and different factors related to these buildings, such as their capacity, energy consumption, etc. The creators of the dataset directly state that this dataset is great for linear regression problems, providing both test and training models. With the test dataset, you can check if your prediction matches the prediction from the training dataset based on certain values. It allows us to validate our learned model based on the training dataset and choose the best model with the most optimal hyperparameters/coefficients. 
The training dataset contains information about 1000 different buildings, while the test dataset - 100 buildings.
### Features:
- `Building Type`
- `Square Footage`
- `Number of Occupants`
- `Appliances Used`
- `Average Temperature`
- `Day of Week`
- `Energy Consumption`

---
### Possible Target Variables for Classification Problems:
- `Building Type`
- `Square Footage`
### Possible Target Variables for Regression Problems:
- **`Energy Consumption`**
- `Appliances Used`
- `Average Temperature`

---
# Real-world relevance or application of the problem:
- The dataset can provide very useful information for construction and architectural engineers and designers. Building type, square footage, and number of occupants can give important details about how many appliances in the building should be used and what the expected energy consumption for such a building is. Moreover, this can be helpful in efficiency planning. People responsible for construction might provide a specific optimal number of appliances for the building to limit energy consumption, making a positive impact on the environment.
- Prediction of the day of the week might show a specific trend in which energy consumption or average temperature is different on weekdays than during weekends.
- Businesses and utility companies can automatically predict what type a specific building it is based on square footage, number of occupants, appliances used, and energy consumption, leading to accurate energy billing.

---
# Potential challenges:
- Some buildings can be built from different materials, a feature that is not included in the dataset) causing the energy consumption or average temperature to be different than our prediction. 
- Buildings may be more densely populated than others, which may result in incorrect predictions of building type
- Since there might be different numbers of different building types, data can be imbalanced, and prediction can be more in favor of a specific type of building, which might be incorrect.
- External factors such as environment, building placement, country, and residents' behavior are not included in the dataset. However, they can significantly affect energy consumption, average temperature, and the appliances used.

---
# Reflection:
The analysis of energy consumption can be very significant for the future of construction, architectural, and power companies, which need to plan accordingly multiple factors to limit expenses while at the same time providing the best customer satisfaction possible. Moreover, building regulatory agencies can automatically assign building types based on the details about these buildings. This unequivocally positively impacts not only the different companies, homeowners, and building residents, but also the environment. Understanding energy demand patterns is very important in today's world when we face global environmental challenges such as greenhouse gas emissions or depletion of non-renewable resources. All things considered, the usage of machine learning on this dataset can be very useful for different industries. It is important to train this model often and the addition of other attributes can make this model even more accurate.
