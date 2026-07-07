### Cambiamenti
* Aggiustato codice data_gathering
* Organizzato codice in simple e complex models
* Implemetato optuna per hypertuning automatizzato
* Implementato un inizio di Bayesan Optimization
* Aggiunta deviazione standard per ognuno dei parametri
* Diviso il costo per istanza in due parametri (costo totale - istanze completate): Più semplice da imparare e più utile

### Risultati (per ora)
MSE: 0.4 (iniziale) $\rightarrow$ 0.25 (data gathering migliorato) $\rightarrow$ 0.20 (hypertuning) $\rightarrow$ 0.15 (split costo per istanza) $\rightarrow$ 0.10 (Modello più complesso) $\rightarrow$ ?? (hypertuning 2) (per adesso sembra non cambiare molto, ma servono più dati (0.36 di dropout...))

### TODO
* Test con diverso numero di simuazioni
* Progettazione e implementazione sezione ML su client DTLog
* Aggiungere un "punteggio divergenza" che indichi quanto una richiesta di kpis (cost, duration) si allontana dalle simulazioni (e quindi non è possibile da raggiungere (e.g. cost = 0, duration = 10000s))


