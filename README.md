# NLP Chatbot Model Training

This project is designed to build and train a chatbot using Natural Language Processing (NLP) techniques. The chatbot model is implemented using PyTorch and is trained on custom intents data.

## Project Structure

- **nltk_utilities.py**: Contains utility functions for NLP tasks like tokenization, stemming, and creating a bag of words.
- **training.py**: The main script that loads the data, preprocesses it, defines the model, and trains it.
- **myModel.py**: Where the model class is setup
- **main.py**: Contains the main loop for the chatbot to run
- **data.json**: Contains different intents which on we train the model, this can be extended to make the model generate accurate answers
- **modelData.pth**: The weights of the model and its different parameters are saved here

## Requirements

- Python 3.x
- PyTorch
- NLTK
- NumPy
- JSON

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**:
   - Ensure you have a `data.json` file that contains your intents and patterns.

2. **Run Training**:
   - Execute `training.py` to start the training process:
   ```bash
   python training.py
   ```

3. **Model Saving**:
   - The trained model is saved as `modelData.pth`.

4. **Running the Code**:
   - run `python main.py` to start chatting and type `exit` to abort the program


## Notes

- Ensure that the `nltk` package is properly set up by uncommenting the `nltk.download('punkt')` line if necessary.
- The script is configured to use GPU if available; otherwise, it will default to CPU.

## License

This project is licensed under the MIT License.
