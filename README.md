# Huber-GMM
Matlab codes for Huber GMM for the spatial autoregressive (SAR) model in Tuo Liu, Xingbai Xu, Lung-fei Lee, Yingdan Mei (2025). You need the global optimization toolbox to run the program. If you do not have it, you can modify the codes slightly to use "fminunc" or "fminsearch" for optimization in Matlab.

This program is robust to outliers and conditional heteroskedasticity. Even when the data are i.i.d. normally distributed, it loses only little efficiency, as demonstrated by simulation studies. 

I have also attached the code for QMLE of the SAR model. 
