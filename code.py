import pandas as pd #manipuare si analiza
import numpy as np #operatii matematice si manipulare
from sklearn import metrics 
from sklearn.model_selection import train_test_split #impartirea datelor
from sklearn.ensemble import RandomForestClassifier 

df = pd.read_csv(r'C:/Users/User/Frogs_MFCCs.csv') #citire dintr un fisier CSV

print(df.to_string()) #afisarea intregii tabele(dataframe)

data = pd.DataFrame(df) #se creeaza un tabel numit data pe baza df

X = data[['MFCCs_ 1','MFCCs_ 2','MFCCs_ 3','MFCCs_ 4','MFCCs_ 5','MFCCs_ 6','MFCCs_ 7',
        'MFCCs_ 8','MFCCs_ 9','MFCCs_10','MFCCs_11','MFCCs_12','MFCCs_13',
        'MFCCs_14','MFCCs_15','MFCCs_16','MFCCs_17','MFCCs_18','MFCCs_19',
        'MFCCs_20','MFCCs_21','MFCCs_22']]
y = data["RecordID"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = True)

c = np.array([0.25, 0.5, 0.85]) #procentul in-bag
d = np.array([0.1, 0.5, 0.8]) #numarul de dimensiuni din nod

for a in c:
    for b in d:
        clf = RandomForestClassifier(n_estimators = 10, max_samples = a, max_features = b)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        print('\n')

#se vor parcurge diferitele valori din samples si features iar pentru fiecare
#combinatie a lor se creeaza un nou clasificator care sa se antreneze pe datele
#respective
#apoi sunt facute predictii pe datele de test si se afiseaza acuratetea
