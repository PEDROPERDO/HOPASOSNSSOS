import stopwords, gdown, os
import numpy as np
import tensorflow as tf

inload = lambda url : gdown.download(url=url, quiet=True)

tokenurl = "https://drive.google.com/uc?id=1cNkkJ5bzpPaMTUreS5fnl2SbsdqmX7K9"
if not os.path.isfile("tokener.npy"):
  inload(tokenurl)

tokener = np.load('tokener.npy')

modelurl = "https://drive.google.com/uc?id=189DS2Cz19FJFOv5wWXoT0TuJbVDq_HFA"
if not os.path.isfile("model.h5"):
  inload(modelurl)

tokeni = tf.keras.layers.TextVectorization(
  pad_to_max_tokens=True, 
  output_sequence_length=180,
  max_tokens = 15000
)

tokeni.adapt(tokener)

def controller(sentence : str):
  unlist = stopwords.get_stopwords("english")
  sentence = sentence.lower().split()
  result = [i for i in sentence if i not in unlist]
  result = " ".join(result)
  result = tokeni(result).numpy()
  return result

def logits(sentence):
  model = tf.keras.models.load_model("model.h5", compile=False)
  test = sentence.reshape(1,-1)
  result = round(model.predict(test, verbose=0)[0][0])
  return result
