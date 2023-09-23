# CO2_H2O_prediction
### Introduction
We are aiming to improve the prediction error of carbon dioxide and water concentration in silicate melts. 
### Data Description
The algorithm was trained on experiments that have the concentrations of various compounds documented, such as SiO2, TiO2, etc. There is an example input template in the ```Data``` folder.
### Methodology
We are buidling a weighted linear regression based model in Python with ```statsmodels``` module.
### Conclusion
The algorithm is able to predict carbon dioxide concentration with 5% error in terms of ppm, and 10% for water concentration.
