# AI/ML Engineering Portfolio

A comprehensive collection of code examples and Jupyter notebooks demonstrating key concepts and practical implementations in Artificial Intelligence and Machine Learning. This portfolio showcases hands-on experience with PyTorch and Keras frameworks, covering fundamental concepts to advanced techniques including deep learning, reinforcement learning, and transformer architectures.



## Project Examples

### `PyTorch/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| [`BasicTensorExamples.ipynb`](3%20-%20PyTorch%20examples/BasicTensorExamples.ipynb) | Introduction to PyTorch tensors, data types, and basic operations | ![Tensor Examples](3%20-%20PyTorch%20examples/ReadmeImages/TensorExamples.png) |
| [`HandwritingClassifier.ipynb`](3%20-%20PyTorch%20examples/HandwritingClassifier.ipynb) | Multi-class neural network with MNIST to classify hand-written digits using cross entropy function as the criterion for loss | ![MNIST Writing Detection](3%20-%20PyTorch%20examples/ReadmeImages/MNISTWritingDetection.png) |

### `Keras/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| [`SatelliteClassification.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/SatelliteImageryClassifier/SatelliteClassification.ipynb) | Training a CNN classifier to differentiate and classify agricultural vs. non-agricultural land | ![Satellite Imagery](2%20-%20Keras%20examples/Deep%20Learning/SatelliteImageryClassifier/READMEImages/AgriImagery.png) |
| [`TextDataHandling.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/Building%20Advanced%20Transformers/TextDataHandling.ipynb) | Text preprocessing with TensorFlow TextVectorization layer | ![Text Analysis](3%20-%20PyTorch%20examples/ReadmeImages/Text%20Analysis.webp) |
| [`TimeSeriesPrediction.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/Building%20Advanced%20Transformers/TimeSeriesPrediction.ipynb) | Advanced transformer-based time series forecasting of stock market data | ![Stock Market Prediction](2%20-%20Keras%20examples/Deep%20Learning/Building%20Advanced%20Transformers/ReadmeImages/StockMarketPrediction.png) |
| [`CancerClassification.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/CancerClassification/CancerClassification.ipynb) | Deep learning model for breast cancer classification using histopathological images | ![Breast Cancer Classifier](2%20-%20Keras%20examples/Deep%20Learning/CancerClassification/ReadmeImage/BreastCancerClassifier.png) |
| [`CustomTrainingLoopExample.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/Keras%20Techniques/CustomTrainingLoopExample.ipynb) | Implementation of custom training loops with callbacks | |
| [`HyperparameterTuningWithKerasTunerExample.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/Keras%20Techniques/HyperparameterTuningWithKerasTunerExample.ipynb) | Automated hyperparameter optimization using Keras Tuner | |
| [`ModelOptimizationTechniques.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/Keras%20Techniques/ModelOptimizationTechniques.ipynb) | Model optimization and performance enhancement strategies | |
| [`RNN_For_TimeSeries_Predictions.ipynb`](2%20-%20Keras%20examples/Deep%20Learning/RNNs/RNN_For_TimeSeries_Predictions.ipynb) | Recurrent Neural Networks for time series forecasting | ![TimeSeries Prediction](3%20-%20PyTorch%20examples/ReadmeImages/TimeSeries%20Prediction.png) |
| [`OpenAIGymRLExample.ipynb`](2%20-%20Keras%20examples/Reinforcement%20Learning/OpenAIGymRLExample.ipynb) | Deep Q-Network (DQN) implementation with CartPole environment | ![OpenAI Gym](2%20-%20Keras%20examples/Reinforcement%20Learning/ReadmeImages/OpenAIGym.png) |

### `Generative AI and LLM examples/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| [`DocumentClassifier.ipynb`](1%20-%20Generative%20AI%20and%20LLM%20examples/DocumentClassifier.ipynb) | Neural network text classifier using PyTorch and TorchText for AG_NEWS dataset categorization with EmbeddingBag architecture | ![Document Classifier](1%20-%20Generative%20AI%20and%20LLM%20examples/ReadmeImages/DocClassifier.png) |
| [`MachineTranslation.ipynb`](1%20-%20Generative%20AI%20and%20LLM%20examples/MachineTranslation.ipynb) | Sequence-to-sequence RNN model for German-to-English translation using PyTorch, Multi30K dataset, and encoder-decoder architecture with LSTM | ![Translation](3%20-%20PyTorch%20examples/ReadmeImages/Translation.png) |

### `General Statistics/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| [`PlottingExamples.ipynb`](4-%20General%20statistics/PlottingExamples.ipynb) | MatPlotLib plotting examples demonstrating histograms, pie charts, and dot plots for descriptive statistics and data visualization | ![Scatter Plot](4-%20General%20statistics/ReadMeImages/scatterplt.png) ![Whisker Plot](4-%20General%20statistics/ReadMeImages/whiskerPlot.png) |

### `Concurrent Programming and CUDA/`
| File/Directory | Description | Screenshot |
|----------------|-------------|------------|
| [`CUDA/vector_add/`](5-%20Concurrent%20Programming/CUDA/vector_add/) | CUDA kernel implementation for parallel vector addition on GPU | |
| [`CUDA/matrix_multiply/`](5-%20Concurrent%20Programming/CUDA/matrix_multiply/) | GPU-accelerated matrix multiplication using CUDA | |
| [`concurrency_pattern_examples/MapReduce/wordcount.py`](5-%20Concurrent%20Programming/concurrency_pattern_examples/MapReduce/wordcount.py) | MapReduce pattern implementation for distributed word counting | |
| [`concurrency_pattern_examples/Repository/repository_pattern.py`](5-%20Concurrent%20Programming/concurrency_pattern_examples/Repository/repository_pattern.py) | Repository pattern for thread-safe data access | |

### `Agentic AI/`
| Notebook | Description | Screenshot |
|----------|-------------|------------|
| [`StarterAgentNotebook.ipynb`](6-%20Agent%20Examples/StarterAgentNotebook.ipynb) | Foundational agent implementation and development patterns | |
| [`Environment and LiteLLM Setup.ipynb`](6-%20Agent%20Examples/Environment%20and%20LiteLLM%20Setup.ipynb) | Configuration for LiteLLM integration and agent tooling | |

*(Under development)*

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
- **Machine Translation**: Sequence-to-sequence RNN models with encoder-decoder architecture for language translation
- **LSTM Networks**: Bidirectional processing for context understanding and sequence generation
- **Teacher Forcing**: Training strategy balancing ground truth and model predictions during sequence generation

### General Statistics
- **Data Visualization**: MatPlotLib plotting techniques for descriptive statistics
- **Histogram Analysis**: Frequency distribution visualization for continuous data
- **Categorical Data Visualization**: Pie charts and dot plots for comparing categories
- **Statistical Plotting**: Foundation techniques for data exploration and analysis

## Getting Started

Each notebook is self-contained with installation instructions and can be run independently. The notebooks include:
- Environment setup and dependency installation
- Step-by-step implementation explanations
- Practical examples with real datasets
- Performance monitoring and visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
