# COMP 6630 Machine Learning Final Project
Final project for COMP 6630 Machine Learning. This class was taught by Professor Bo Liu at Auburn University.


Link for GloVe dataset

https://nlp.stanford.edu/projects/glove/

## Contents

- **/data** - contains data that was used to train all of the models, as well as the Python script used to scrape the data. The data is organized by genere.
- **/gan** - contains source code for the GAN model, along with saved model states of each nth training iteration and generated song results
- **/lstmGAN** - contains source code for the GAN model with the LSTM generator, along with saved model states of each nth training teration and generated song results
- **/lyric_outputs** - contains genreated lyrics for the LSTM baseline model with and without using the GloVe dataset enhancer
- **/provided** - documents provided for the assignment from the class
- **.gitignore** - git file for ignoring certain local files when pushing changes
- **LICENSE** - license for using this project 
- **README.md** - this file that you are reading
- **gan-generator-discriminator-loss.png** - training loss results image for the basic GAN model
- **lstm-medium-glove-train-accuracy.png** - training accuracy results image for the LSTM model using the medium GloVe dataset
- **lstm-medium-glove-train-loss.png** - training loss results image for the LSTM model using the medium GloVe dataset
- **lstm-none-glove-train-accuracy.png** - training accuracy results image for the LSTM model using no GloVe dataset
- **lstm-none-glove-train-loss.png** - training loss results image for the LSTM model using no GloVe dataset
- **lstm.py** - source code for the LSTM baseline model
- **requirements.txt** - Python dependencies for this project
