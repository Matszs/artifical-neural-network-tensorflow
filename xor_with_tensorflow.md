![TensorFlow Logo](https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688)
# XOR MET TENSORFLOW
Auteurs: Mats Otten en Patrick Hendriks

Machine learning wordt vaak toegepast op problemen waar mensen geen directe logische oplossing voor lijken te hebben.
Problemen oplossen door middel van machine learning is niet altijd de meest handige keus wanneer er ook andere oplossingen te vinden zijn.
Dit komt namelijk doordat er vaak veel van de computer danwel grafische processor geeist wordt. Ook moet er altijd een trainingsset beschikbaar zijn om de machine een patroon te leren herkennen.
In dit document gaan we een simpele booleaanse logische gate proberen in een neuraal netwerk na te bouwen. We doen dit voor de XOR operatie.

## A.I. Intro
Kunstmatige intelligentie is gemodelleerd naar menselijke intelligentie. Het verbaast dan ook niet dat alle terminilogie zoals neuronen gewoonweg overgenomen zijn van wat wij mensen weten van menselijke en dierlijke intelligentie.
Om deze neuronen na te bootsen is er met de booleaanse computerlogica veel van deze schakelingen en berekeningen nodig om een enkele neuron te simuleren. Het mag dan duidelijk zijn dat complexe operaties ook veel meer neuronen en dus ook veel meer computerkracht nodig hebben.
Hoewel het process met een grote stap versneld kan worden met Grafische processing units van videokaarten, is dit slechts een optie waarbij veel berekingen paralel aan elkaar berekend kunnen worden. En hoewel dit vaak het geval is,
is het slechts voor het aantal neuronen gelijktijdig in een layer. En een complex machine learning programma bestaat vaak ook uit meerdere layers. 

Het probleem wat kunstmatige intelligentie kan oplossen is vaak eenzijdig. Er is een grote set aan data nodig om een machine een overeenkomst 

### TensorFlow
TensorFlow is een open source machine learning framework. Er zijn ook nog andere frameworks beschikbaar om machine learning problemen mee op te lossen zoals Theano maar TensorFlow is de meest bekende en daardoor zul je waarschijnlijk ook het snelste meer en betere uitleg hebben over tensorflow dan anderen.

#### Lagen en gewichten
Om te beginnen met het opzetten van je neurale netwerk moet je natuurlijk weten hoeveel input neuronen je hebt, hoeveel hidden layers en hidden neuronen en tot slot het aantal output neuronen.

Dan, voordat je het neurale netwerk traint, moet je initiële waardes geven voor de gewichten van iedere “lijn”. Gewichten voor normale connecties tussen neuronen noemen we theta (θ). In TensorFlow zijn dit zogenoemde matrixen die alle gewichten van connecties tussen twee lagen bevat.
 

De theta matrix van bovenstaande neuraal netwerk, met bias neuronen (groen) ziet er dus zo uit voor laag 1 (rechts)

	
Theta1 = gewichten van input layer naar hidden layer. 
Theta2 = gewichten van hidden layer naar output layer

De gewichten starten met willekeurig getal tussen -1 en 1. Deze waardes worden bij iedere iteratie geüpdatet om de error ten opzichte van het gewenste resultaat te verminderen. Let op dat wij in TensorFlow bij de theta de bias connecties niet meenemen. Deze behandelen we in een aparte matrix.

Bias1 = gewicht/waarde van de bias van input layer naar hidden layer
Bias2 = gewicht/waarde van de bias van hidden layer naar output layer

De gewichten van de bias connecties geven we initieel een 0. Ook deze worden tijdens het leerproces veranderd naar de waardes die het beste resultaat geeft.

#### Matrixen
Zoals eerder gezegd maken we in TensorFlow gebruik van matrixen. Dit zijn eigenlijk één- of meerdimensionale arrays. Eendimensionale matrixen worden ook wel vectors genoemd. Het voordeel van het gebruiken van Matrixen is dat je in TensorFlow op een gemakkelijke manier deze met elkaar kunt vermenigvuldigen, of natuurlijk andere wiskundige operaties mee kunt uitvoeren.

Het vermenigvuldigen van input matrix x_ met de bijbehorende gewichten:
```python
tf.matmul(x_, Theta1)
```
#### Variabelen
In TensorFlow heb je twee soorten variabelen. De eerste soort variabelen heten placeholder en worden voornamelijk gebruikt voor je input (trainings-) data. De naam placeholder komt voor uit het feit dat ze niet gelijk geïnitialiseerd worden, maar tijdens het trainen iedere keer een nieuwe (of dezelfde) trainingsdata krijgen toegewezen. Hieronder maken we een placeholder aan van het type float32 en een grootte van een matrix van 4x2.
```python
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
```
Als we de placeholder willen vullen tijdens het uitvoeren van de code, doen we dat zo:

```python
sess.run(cost, feed_dict={x_:[[0,0],[0,1],[1,0],[1,1]]})
```
Vervolgens zijn er de overige variabelen die de gewichten van de theta en bias bevatten. Deze moeten wel direct een initiële waarde krijgen, die tijdens het trainen worden geüpdatet. Hieronder maken we de variabele Theta1 aan met een matrix van 2x2, die elk een random waarde krijgen tussen de -1 en 1.
```python
Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
```
## Aanleren van XOR

|  Input     |  |Output   |
|---|---|---|
| X1  | X2  |  Y |
|  0 | 0  |  0 |
|  0 | 1  |  1 |
|  1 | 0  |  1 |
|  1 | 1  |  0 |

In dit practicum is het de bedoeling dat we in Python een applicatie schrijven om met TensorFlow een XOR te “leren”. Een XOR-poort is een booleaanse operator. Hiernaast zie je de waarheidstabel van deze poort met twee input waardes. De waarde van de output is 1 als een van de ingangen 1 is maar niet beide.

Voor dit practicum gaan we 2 input waardes gebruiken en 1 output. In TensorFlow wordt de input een matrix, zoals eerder beschreven, met alle waardes en de output wordt een matrix met verwachte 4 waardes.

## Installeren TensorFlow
Voor het installeren van tensorflow verwijs ik je naar de uitleg van [intro to tensorflow](intro_to_tensorflow.md)

Om te testen of TensorFlow succesvol is geïnstalleerd, run dit python scriptje:

```python

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
Hello, TensorFlow!
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
42
```
## Uitleg Python Tensorflow code
Als TensorFlow is geïnstalleerd kan je de volledige code runnen. Download en run: [https://github.com/StephenOman/TensorFlowExamples/tree/master/xor](https://github.com/StephenOman/TensorFlowExamples/tree/master/xor%20nn) 

We zullen hieronder de belangrijkste regels toelichten zodat het enigszins duidelijk wordt wat er gebeurd.

Als eerste moeten we TensorFlow in ons programma laden:

```python
import tensorflow as tf
```
Daarna maken we placeholders voor de trainingsdata, de input (x) en de output (y):

```python
x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")
```
Met de shape parameter geven we aan welke dimensies de data die we erin gaan stoppen heeft.

Vervolgens maken we de variabelen aan die de gewichten gaan onthouden. We geven ze direct een random array/matrix met waardes tussen -1 en 1:
```python
Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name="Theta1")
Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name="Theta2")
```
De gewichten van de bias nodes/neuronen gaan we apart aanmaken, maar zijn ook variabelen Deze geven we de waarde 0:
```python
Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")
```
Vervolgens moeten we waardes van de output van de hidden layer bereken. Dit doen we door de waardes van de inputs te vermenigvuldigen met bijbehorende gewichten, vervolgens de bias erbij optellen en deze in de sigmoid functie te stoppen:
```python
A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Let op dat we hier alleen de “opzet” van de variabele A2 creëren, pas tijdens het runnen van de code wordt hiervoor de waarde berekend. (Zie sess.run()).
```
Hierna moeten we hetzelfde doen bij de output layer. Om de “voorspelling” te krijgen. We gebruiken nu dus de output van laag 2 (hidden layer) en de bijbehorende bias en gewichten:

Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)
Nu moeten we met de cost functie gaan kijken hoe ver de voorspelling van de werkelijke output vandaag zit:

```python
cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
        ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
```
De volgende stap is om alle gewichten zodanig te veranderen dat de uitkomst van de cost functie zo laag mogelijk is. Hier heeft TensorFlow een ingebouwde functie voor:

```python
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
```
Dit commando zegt dat we de Gradient Descent Optimizer als trainingsalgoritme gaan gebruiken. De 0.01 betekend hoe veel de gewichten veranderd mogen worden. Voor dit experiment is 0.01 een goede waarde.

Nu we het netwerk hebben opgezet moeten we wat initialisaties doen:
```python
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
```
TensorFlow maakt het model van het neurale network binnen een sessie. Nu we de sessie hebben aangemaakt kunnen we de placeholders gaan initialiseren en het netwerk trainen:
```python
for i in range(100000):
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
```
Met feed_dict laden we de waardes die we meegeven in de placeholders. Een iteratie in het netwerk wordt ook wel een Epoch genoemd. We laten het netwerk 100.000 keer leren. Om te zien wat er tijdens de iteraties met de variabelen gebeurd printen we elke 1000 iteraties alle waardes van de variabelen:
```python
if i % 1000 == 0:
        print('Epoch ', i)
        print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('Theta1 ', sess.run(Theta1))
        print('Bias1 ', sess.run(Bias1))
        print('Theta2 ', sess.run(Theta2))
        print('Bias2 ', sess.run(Bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
```
TensorFlow heft ook een tool genaamd TensorBoard. Hier kan je visuele weergave van je neurale netwerk bekijken, in de vorm van een directed graph. Hiervoor hebben we bij de initialisatie de log toegevoegd:
```python
writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph_def)
En nu kan je TensorBoard starten met:
```
tensorboard --logdir=/path/to/your/log/file/folder
Je krijgt dan zoiets te zien: 
