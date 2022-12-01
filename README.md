# Deeplog (end to end anomaly detection)

- **Input**: raw logs(log/text) in `input` folder
- **output**: csv structuded files with marked anomalies in `output` folder 

Structure
Required packages: 

`Baseline` contains first cell which installs reuire packages automatically, used for parsing, training logs and testing logs. 
> while parsing keep `options['logkeys'] = True` during training to keep logkeys and while predicting keep it `False`
> logparse
Contains `Spel` and `Drain` two parsing algorithms default is `Drain`

> loganomaly
Pre-processing, model, Training and Testing files. 

## Deeplog Architecture:
![image](https://user-images.githubusercontent.com/56737996/205067830-1a8060f9-cfe2-4ba7-9ce1-ba3400b2f953.png)



REFERENCE Deeplog Paper:
https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
