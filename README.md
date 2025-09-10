# AI/ML Engineering Portfolio

A comprehensive collection of Jupyter notebooks demonstrating key concepts and practical implementations in Artificial Intelligence and Machine Learning. This portfolio showcases hands-on experience with PyTorch and Keras frameworks, covering fundamental concepts to advanced techniques including deep learning, reinforcement learning, and transformer architectures.



## Project Structure

### `PyTorch/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| `BasicTensorExamples.ipynb` | Introduction to PyTorch tensors, data types, and basic operations | ![Tensor Examples](PyTorch%20examples/ReadmeImages/TensorExamples.png) |
| `HandwritingClassifier.ipynb` | Multi-class neural network with MNIST to classify hand-written digits using cross entropy function as the criterion for loss | ![MNIST Writing Detection](PyTorch%20examples/ReadmeImages/MNISTWritingDetection.png) |

### `Keras/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| `SatelliteClassification.ipynb` | Training a CNN classifier to differentiate and classify agricultural vs. non-agricultural land | ![Satellite Imagery](Keras%20examples/Deep%20Learning/SatelliteImageryClassifier/READMEImages/AgriImagery.png) |
| `TextDataHandling.ipynb` | Text preprocessing with TensorFlow TextVectorization layer | |
| `TimeSeriesPrediction.ipynb` | Advanced transformer-based time series forecasting of stock market data | ![Stock Market Prediction](Keras%20examples/Deep%20Learning/Building%20Advanced%20Transformers/ReadmeImages/StockMarketPrediction.png) |
| `CancerClassification.ipynb` | Deep learning model for breast cancer classification using histopathological images | ![Breast Cancer Classifier](Keras%20examples/Deep%20Learning/CancerClassification/ReadmeImage/BreastCancerClassifier.png) |
| `CustomTrainingLoopExample.ipynb` | Implementation of custom training loops with callbacks | |
| `HyperparameterTuningWithKerasTunerExample.ipynb` | Automated hyperparameter optimization using Keras Tuner | |
| `ModelOptimizationTechniques.ipynb` | Model optimization and performance enhancement strategies | |
| `RNN_For_TimeSeries_Predictions.ipynb` | Recurrent Neural Networks for time series forecasting | |
| `OpenAIGymRLExample.ipynb` | Deep Q-Network (DQN) implementation with CartPole environment | ![OpenAI Gym](Keras%20examples/Reinforcement%20Learning/ReadmeImages/OpenAIGym.png) |

### `Generative AI and LLM examples/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| `DocumentClassifier.ipynb` | Neural network text classifier using PyTorch and TorchText for AG_NEWS dataset categorization with EmbeddingBag architecture | ![Document Classifier](Generative%20AI%20and%20LLM%20examples/ReadmeImages/DocClassifier.png) |

## Key Concepts Covered

### PyTorch
- Tensor creation and manipulation
- Data type management and conversion
- Basic tensor operations and indexing
- **Handwriting Classification**: Multi-class neural network for MNIST digit recognition using cross-entropy loss

### Keras Deep Learning
- **Advanced Transformers**: Text vectorization, sequence processing, time series prediction
- **Medical Image Classification**: Breast cancer classification using histopathological images
- **Custom Training**: Manual training loops, custom callbacks, metric monitoring
- **Model Optimization**: Hyperparameter tuning, performance optimization techniques
- **RNNs**: Recurrent neural networks for sequential data processing

### Reinforcement Learning
- Deep Q-Network (DQN) architecture
- OpenAI Gymnasium environment integration
- Experience replay and target network implementation
- Exploration vs exploitation strategies

### Generative AI and LLM
- **Document Classification**: Text preprocessing, tokenization, and vocabulary building with TorchText
- **Neural Text Processing**: EmbeddingBag architecture for efficient text representation
- **Multi-class Classification**: Cross-entropy loss optimization for news article categorization
- **Real-world NLP Pipeline**: End-to-end implementation from data loading to model inference

## Getting Started

Each notebook is self-contained with installation instructions and can be run independently. The notebooks include:
- Environment setup and dependency installation
- Step-by-step implementation explanations
- Practical examples with real datasets
- Performance monitoring and visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
