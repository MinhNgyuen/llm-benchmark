# LLM Benchmark

This tool allows you to easily benchmark the inference performance of various Large Language Models (LLMs) on your local machine. Currently we only support testing Ollama llms

## Getting Started

Follow these instructions to set up and run benchmarks on your system.

### Prerequisites

- Python 3.6+

### Installation

1. **Clone this repository**

   Open a terminal or powershell and run:

   ```bash
   git clone https://github.com/yourusername/llm-benchmark.git
   cd llm-benchmark
   ```

2. **Create a virtual environment**

   - **Windows**

     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - **Linux/macOS**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Serve the Ollama model**

   Before running benchmarks, make sure your Ollama model server is running:

   ```bash
   ollama serve
   ```

### Running Benchmarks

To run benchmarks, use the `benchmark.py` script with the desired command line arguments:

```bash
python benchmark.py --verbose --prompts "What is the sky blue?" "Write a report on the financials of Nvidia"
```

#### Command Line Arguments

- `--verbose`: Increase output verbosity for more detailed benchmark logs.
- `--skip-models`: Specify a list of model names to skip during the benchmark. Separate multiple models with spaces.
- `--prompts`: Provide custom prompts to use for benchmarking. Separate multiple prompts with spaces.

#### Examples

- **Run with default prompts in verbose mode**

  ```bash
  python benchmark.py --verbose
  ```

- **Run with custom prompts**

  ```bash
  python benchmark.py --prompts "Custom prompt 1" "Custom prompt 2"
  ```

- **Skip specific models**

  ```bash
  python benchmark.py --skip-models model1 model2
  ```

## Contributing

We welcome contributions! Please feel free to submit a pull request or create an issue if you have suggestions for improvements or have identified bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
