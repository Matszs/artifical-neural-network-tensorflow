# Artificial Neural Network maken met TensorFlow

Dit document beschrijft de basis van het werken met Tensorflow. Met deze handleiding leer je de
beginselen van het werken met Tensorflow in Python. 

### Vereisten
Er wordt vanuit gegaan dat de lezer reeds over deze kennis beschikt:
* De programmeertaal Python.
* Basiskennis van artificial neural networks.

### Leerdoelen
Aan het eind van deze opdracht kun je de volgende dingen:
* Uitleggen wat TensorFlow is.
* De basis van TensorFlow begrijpen en uitleggen.
* Een simpele artificial neural netwerk maken met TensorFlow.


### Wat is TensorFlow?
TensorFlow is een open source software library die data flow graphs gebruikt voor numerieke berekeningen. Het is in 
november 2015 open source gemaakt door Google. De library wordt onder andere gebruikt voor natural language processing, 
artificial intelligence, computer vision en predictive analysis. TensorFlow is een schaalbaar systeem: het kan zowel op 
een enkele smartphone draaien als op duizenden computers in een datacenter. Naast dat TensorFlow berekeningen op meerdere 
computers kan uitvoeren kan het ook berekeningen over meerdere CPUs en GPUs verdelen.

De library heeft een API voor de programmeertalen Python en C++. Daarnaast zijn er experimentele APIs voor Go en Java. 
Tijdens deze opdracht gaan we met de Python API aan de slag.

##### Numerieke berekeningen
In plaats van dat Google TensorFlow een library voor machine learning noemt, gebruikt Google de bredere term “numerieke 
berekeningen”. Het primaire doel van TensorFlow is niet het leveren van out of the box machine learning oplossingen. In 
plaats daarvan levert TensorFlow een breed scala aan klassen en functies die gebruikt kunnen worden om vanaf nul 
wiskundige modellen te maken.

##### Data flow graphs
Een belangrijke datastructuur binnen TensorFlow is de data flow graph. Een data flow graph is een gerichte graaf met 
edges en nodes. De nodes in een data flow graph representeren berekeningen (ops of operations genoemd), terwijl de edges 
multidimensionale data arrays (tensors) representeren. Een op neemt als input 0 of meerdere tensors, voert een 
berekening uit en heeft als output 0 of meerdere tensors. Een data flow graph is een representatie van berekeningen die 
plaatsvinden binnen een sessie.
Het voordeel van het definiëren van een graaf is dat de berekeningen makkelijker te distribueren zijn. De graaf hoeft 
slechts opgesplitst te worden en verdeeld te worden over meerdere machines.
Een TensorFlow programma bestaat over het algemeen uit twee fases: een construction phase en een execution phase. In de 
construction phase wordt er een een data flow graph geconstrueerd terwijl ops in de execution phase binnen een sessie 
worden uitgevoerd.
Eerste stappen

Voordat we met TensorFlow kunnen beginnen moet je het installeren. Zorg ervoor dat je python 2 en pip reeds op je machine hebt geïnstalleerd. Voer het volgende commando uit: pip install tensorflow.
Voer vervolgens het commando python uit en voer de volgende regels code uit:

<pre>
import tensorflow as tf

hello = tf.constant('Hello world!')
sess = tf.Session()

print sess.run(hello)
</pre>

Als het goed is zie je nu de tekst “Hello world!” op het scherm verschijnen. Zo niet, dan is er wat fout gegaan met de installatie van TensorFlow. Probeer dit eerst op te lossen voordat je verder gaat.
We kunnen met TensorFlow een simpele rekensom uitvoeren:
<pre>
import tensorflow as tf

op1 = tf.constant(5)
op2 = tf.constant(3)

product = tf.mul(op1, op2)

session = tf.Session()
print session.run(product)
session.close()
</pre>

Wat we hier doen is twee ops aanmaken met tf.constant. De ops hebben geen input nodig, maar geven als output een tensor 
die gelijk is aan de waarde die je hebt meegegeven aan de constructor (5 en 3). We maken ook een op aan die twee inputs 
accepteert en als resultaat het product van de vermenigvuldiging geeft. De TensorFlow Python library maakt standaard een 
default graph aan. Daar worden de ops in dit script aan toegevoegd.

We hebben tot nu toe alleen nog maar een data flow graph gedefinieerd, maar nog geen berekeningen uitgevoerd. We maken 
nu een sessie aan waarbinnen de berekeningen plaatsvinden. Als je aan de constructor van Session geen graph geeft dan 
wordt de sessie aan de default graph gekoppeld. Met session.run voer je de berekeningen uit en door product als argument 
te geven geef je aan dat je het resultaat van de vermenigvuldiging wilt hebben. Tot slot sluiten we de sessie af, zodat 
resources vrijgegeven kunnen worden.
Tensors hebben 3 eigenschappen: rank, shape en data type. De mul op in TensorFlow ondersteunt broadcasting. Broadcasting 
treedt op als je twee tensors met een andere shape met elkaar vermenigvuldigt. Maak een script dat de tafels van 1 tot 
en met 10 print met behulp van twee tensors die je met elkaar vermenigvuldigt. Maak gebruik van broadcasting. Zie de API 
documentatie voor meer informatie (https://www.tensorflow.org/versions/r0.11/api_docs/python/).


### Perceptron
De meest simpele artificial neural network (ANN) is een ANN met slechts 1 perceptron. De toepassingen van zo een ANN zijn 
beperkt, maar om het simpel te houden zullen we hier een ANN met maar 1 perceptron maken. De ANN simuleert een AND gate.
<pre>
import tensorflow as tf

inputs = tf.placeholder(tf.int32, shape=(2))

weights = tf.constant([1, 1])
threshold = tf.constant(2)

perceptron = tf.mul(inputs, weights)
result = tf.reduce_sum(perceptron)

activation_function = tf.greater_equal(result, threshold)

with tf.Session() as sess:
        print sess.run(activation_function, feed_dict={inputs: [0, 0]})
        print sess.run(activation_function, feed_dict={inputs: [1, 0]})
        print sess.run(activation_function, feed_dict={inputs: [0, 1]})
        print sess.run(activation_function, feed_dict={inputs: [1, 1]})
</pre>


We zien een aantal nieuwe ops: placeholder, reduce_sum en greater_equal. Placeholder kun je zien als een 
variabele waar je een waarde aan toe moet kennen. Dit wordt gedaan met de feed_dict parameter van sess.run. 
De op reduce_sum geeft de som van alle elementen en greater_equal geeft 0 als result kleiner is dan threshold 
en 1 als result gelijk aan of groter is dan threshold.

Het is moglijk om een AND en OR gate met 1 perceptron te simuleren. Echter is het niet mogelijk om een XOR gate met maar 
1 perceptron te simuleren. Hoeveel perceptrons heb je hiervoor nodig? Maak met TensorFlow een ANN die een XOR gate 
simuleert.


### Bronnen
* https://www.tensorflow.org/
* https://googleblog.blogspot.nl/2015/11/tensorflow-smarter-machine-learning-for.html
* https://www.tensorflow.org/versions/master/api_docs/
* https://www.oreilly.com/learning/hello-tensorflow
* https://www.tensorflow.org/get_started/basic_usage
* TensorFlow for Machine Intelligence: A Hands-On Introduction to Learning Algorithms (ISBN: 1939902452)

### Versiebeheer



<table>
	<tr>
		<th>Auteur</th>
		<th>Wijzigingen</th>
		<th>Datum</th>
	</tr>
	<tr>
		<td>Joris van der Kallen</td>
		<td>Opzet document</td>
		<td>2016 - 2017</td>
	</tr>
</table>

