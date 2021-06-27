# Finding Wally with YOLO

## Preparando dataset

O dataset fornecido para o treinamento contém 140 imagens no formato `.jpg`. Para cada imagem existe um arquivo
de anotações com o formato `.json`.
Essas anotações estão no padrão gerado pelo software `Labelme`,
disponível em: https://github.com/wkentaro/labelme.
Nesse padrão, são dadas as coordenadas dos 4 pontos da marcação que identifica o objeto,
o que permite até mesmo marcações em formato trapezoidal e losangular.

Entretanto, no padrão esperado pelo framework `darknet`, as marcações precisam ser retangulares e 
sem rotação.
Portanto, os arquivos de anotações precisam ser convertidos para o padrão esperado.

 Onde:

- `<object-class>` - número inteiro do objeto de `0` a `(classes-1)`
- `<x_center> <y_center> <width> <height>` - valores float **relativos** ao comprimento e altura da imagem,
  que pode ser de `(0.0 a 1.0]`
- por examplo: `<x> = <absolute_x> / <image_width>` or `<height> = <absolute_height> / <image_height>`
- atenção: `<x_center> <y_center>` - se referem ao centro do retângulo.

  Por examplo, para `wally_000.jpg` será criado um `wally_000.txt` contendo:

  ```csv
  0 0.682 0.935616 0.25 0.29726
  ```
  
Para realizar a conversão das anotações, basta rodar o código `app_convert_bboxes.py` e passar o diretório
onde se encontram as anotações.

Em seguida, basta utilizar o script ``app_prepare_train_data.py`` para criar uma pasta contendo os arquivos
necessários para o treinamento, além de um arquivo ``train.txt`` que contem uma lista das imagens de treino.
Os arquivos serão salvos em uma pasta na sequência ``data/obj/train``, onde será necessário compactar
manualmente a pasta ``obj`` referente a esse modelo apresentado.
É necessário que tanto o arquivo ``obj.rar`` quanto o ``train.txt`` sejam transferidos para a pasta onde
será feito o treinamento, porém esses arquivos já foram adicionados a pasta do Google Drive 
que é disponibilizada pelo seguinte
[link](https://drive.google.com/drive/folders/1zCHefDD28qBSy5BUazfeKQh0wxfpiX0g?usp=sharing).

## Realizando o treinamento
Para realizar o treinamento será necessário utilizar a plataforma do `Google Colab`.
Para isso, foi disponibilizado um `Jupyter Notebook` contendo todos os passos necessários.

O diretório do Google Drive contém todos os arquivos necessários para o treinamento da rede.

Recomenda-se que esse diretório seja copiado para o diretório principal do `Google Drive` do usuário, 
pois os arquivos utilizados ao longo do treinamento estão sendo referenciados para esse diretório.
No entanto, caso necessário alterar o diretório, basta alterar as linhas de código da seção
`Copying files from Google Drive`.

## Realizando os testes
Para realizar os testes basta utilizar o código ``app_test.py`` e passar os parâmetros referentes aos
arquivos necessários para carregar a rede, como é descrito no próprio código.

Neste repositório já estão disponíveis os arquivos que são utilizados para o treinamento e teste da rede
de detecção, os quais estão contidos na pasta ``models``. Entretanto, é necessário que os pesos sejam
baixados manualmentes pelo usuário, e transferidos, preferivelmente, para o diretório ``models/weights``.

Isso é feito para que os testes possam ser realizados mesmo sem a necessidade de realizar o treinamento,
que deve consumir bastante tempo.


## Conclusões
Infelizmente, devido a falta de técnicas de pré-processamento de imagens, assim como a falta da utilização
de técnicas para treinar com anotações em formatos diferentes ao retangular, os resultados obtidos
pelo detector de objetos foram indesejáveis. 
Pois, além de detectarem os centróides em posições distantes as esperadas, em alguns casos nem mesmo 
foram detectados o objeto esperado.

# SMS Spam Detection

Neste projeto, dentro do arquivo ``spam_detection.py`` é criada uma classe de nome ``SpamDetector``,
a qual contém dois métodos:``prob_spam`` e ``is_spam``, que retornam a probalidade da string de entrada
ser um spam e um booleano se a string de entrada é um spam, respectivamente.

Para testar esses métodos basta passar uma string como argumento para uma das funções de uma instância
da classe ``SpamDetector``. Além disso, é necessário que seja passado o caminho para o arquivo do modelo
pré-treinado.

Para realizar os testes unitários, foi criado um arquivo de nome ``test_spam_detect.py``, contendo 4
testes diferentes, para cada método da classe ``SpamDetector``; assegurando as saídas desses métodos,
e para cada saída esperada, seja ela spam ou ham.

