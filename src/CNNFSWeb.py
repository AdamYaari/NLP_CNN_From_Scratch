"""Usage: CNNFromScratch.py [-h H | --host=H] [-p P | --port=P] [-w W | --weights_file_path=W]
                            [-c C | --config_file_name=C]

Options:
    -h --host=H  host server address [default: localhost]
    -p --port=P  port of host server [default: 8000]
    -w --weights_file_path=W  The absolute name & path for the model's saved weights file [default: ./CNNSavedWeights]
    -c --config_file_name=C  The absolute name & path for the model's test results output file [default: ./CNNTestResults.txt]
"""

import os
import docopt
import pycnn as pc
import CNNFromScratchModule as module
from bottle import run, get, post, request

'''CNNFSWeb is the web intarface to a Pycnn implementation of "Text Understanding from Scratch" paper by Xiang Zhang &
Yann LeCun published in arXiv in April 2016.'''

arguments = docopt.docopt(__doc__)

host = arguments['--host']
port = int(arguments['--port'])
weights_file_path = arguments['--weights_file_path']
config_file_path = arguments['--config_file_name']

id_to_label_dic = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


'''Returns an html format of the login screen with a text box to insert article to be classified and
                a "submit" button to send for examination.

:returns: an html format of the login screen with explanation of the page essence, text box and a "submit" button.
:rtype: string
'''
@get('/CNNFromScratch')
def login():
    return '''
        <form action="/CNNFromScratch" method="post">
            <p><b><font size='6'>NLP CNN From Scratch Website</font></b></p>
            <p>This web page is classifying articles to 4 categories (World, Sports, Business & Sci/Tech) according to
            <i>\'Text Understanding from Scratch\'</i> article, based on a 1D NLP convolutional neural network</p>
            <p>Link to original paper from X. Zhang & Y. LeCun:
            <a href="https://arxiv.org/pdf/1502.01710.pdf">Text Understanding from Scratch paper</a></p>
            <br>Please insert article text to the box bellow</br>
            <p>Article: <input name="article" type="text" size="200"/></p>
            <input value="Submit" type="submit"/>
        </form>
    '''

'''Create a new model, load the trained weights from the weights file, run a forward propagation using the input text
        and predict it's category.

:returns: an answer of which category is the most suited to the given text out of the given selection
:rtype: string
'''
@post('/CNNFromScratch')
def do_login():
    article_to_label = request.forms.get('article')
    if not os.path.exists(weights_file_path):
        return "No weights file was found, looked for file {}, in directory {}".format(weights_file_path, os.getcwd())
    if not os.path.exists(config_file_path):
        return "No configuration file was found, looked for file {}, in directory {}".format(weights_file_path, os.getcwd())

    relu_threshold = None
    with open(config_file_path, "r") as config_file:
        for line in config_file:
            splitted_line = line.split(':')
            if len(splitted_line) > 2:
                return "Configuration file does not contain threshold for ReLU activation function"
            if splitted_line[0].strip() == "thresh":
                if len(splitted_line) < 2:
                    return "Configuration file threshold for ReLU activation function is empty"
                relu_threshold = float(splitted_line[1].strip())
                break

    model = pc.Model()
    module.initialize_cnn_model_weights(model, module.cnn_layers_dimensions, module.dense_layers_dimensions)
    model.load(weights_file_path)

    pc.renew_cg()
    predicted_label = module.predict_label(model, article_to_label, module.cnn_layers_dimensions,
                                           module.dense_layers_dimensions, relu_threshold)

    return "Article was classified to category: {}".format(id_to_label_dic[predicted_label])

# host server and port the page will run on
run(host=host, port=port)
