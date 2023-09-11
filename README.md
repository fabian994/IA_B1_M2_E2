# IA_B1_M2_E2

A01367585 | Fabian Gonzalez Vera

En este repositorio se encuentra, una implementacion del algoritmo de Regresion utilizando Random Forest y el Framework de Machine Learning scikit-learn. El fin del algoritmo es determinar la edad, medida en anillo, de los abalones utilizando sus caracteristicas fisicas.

Dentro del archivo etl.ipynb se realizo un EDA sencillo y se probo el modelo, la **implementacion final** del modelo se encuentra dentro del archivo *e2.py*.

Análisis y Reporte sobre el desempeño del modelo se encuentra en el archivo *Análisis_ Reporte_modelo._A01367585.pdf*

## Requisitos

- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2
- scikit-learn==1.3.0
- seaborn==0.12.2
- mlxtend==0.22.0

## Dataset

El data set fue obtenido de la [base de datos](https://archive.ics.uci.edu/dataset/1/abalone) de la universidad de Irving.

Las Variables utilizadas son:
- Sex_I: One hot encoding de la variable Sex
- length: mm
- Diameter: mm
- Height: mm
- Whole_weight: gramos
- Shell_weight: gramos

Variable Objetivo: Rings(entero)

Uno de los arboles con conforman el Random Forest:
![tree](tree.png)
