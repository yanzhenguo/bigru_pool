## bigru_pool
This is the code for our paper 'Gated Recurrent Neural Network with Attention of Important N-grams for Text Classification'

###Prerequisite
python3(keras, numpy)

###How to execute the code
1. Download Glove vector and put all words' embedding in dir /data with file name 'embedding.300d.npy', and all words with file name 'words.pkl'.
2. Execute file src/#dataset_name#/#script_name# with python command. Example:

    ``python src/imdb/gru_ngram_order.py
    ``