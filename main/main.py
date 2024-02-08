# main.py

from src.banchmark import Banchmark

def main():
    nome = input('Digite o nome que gostaria de dar à esta bateria de testes: ')
    bm = Banchmark(nome)

    dataFrames = bm.carrega_datasets()
    print(dataFrames)
    preprocessedDataFrames = bm.preprocessa_datasets(dataFrames)
    print(preprocessedDataFrames)

if __name__ == "__main__":
    main()