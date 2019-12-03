import numpy as np
from scipy.stats import norm

maleW = [180,190,170,165]
maleF = [12,11,12,10]
maleH = [6,5.92,5.58,5.92]
femaleW = [100,150,130,150]
femaleF = [6,8,7,9]
femaleH = [5,5.5,5.42,5.75]
#男性体重，脚长，身高均值
mean_MW = np.mean(maleW)
mean_MF = np.mean(maleF)
mean_MH = np.mean(maleH)
print('男性均值：',mean_MH,mean_MW,mean_MF)
#男性体重，脚长，身高均值
mean_FW = np.mean(femaleW)
mean_FF = np.mean(femaleF)
mean_FH = np.mean(femaleH)
print('女性均值：',mean_FH,mean_FW,mean_FF)
#女性体重，脚长，身高标准差
std_MW = np.std(maleW,ddof = 1)
std_MF = np.std(maleF,ddof = 1)
std_MH = np.std(maleH,ddof = 1)
print('男性方差：',std_MH**2,std_MW**2,std_MF**2)
#男性体重，脚长，身高标准差
std_FW = np.std(femaleW,ddof = 1)
std_FF = np.std(femaleF,ddof = 1)
std_FH = np.std(femaleH,ddof = 1)
print('女性方差：',std_FH**2,std_FW**2,std_FF**2)


P_F = 0.5                                #男性概率
P_M = 0.5                                #女性概率
P_H_F = norm.pdf(6,mean_FH,std_FH)       #男性身高6英尺的概率
P_F_F = norm.pdf(8,mean_FF,std_FF)       #男性脚长8英寸的概率
P_W_F = norm.pdf(130,mean_FW,std_FW)     #男性体重130磅的概率
P_H_M = norm.pdf(6,mean_MH,std_MH)       #女性身高6英尺的概率
P_F_M = norm.pdf(8,mean_MF,std_MF)       #女性脚长8英寸的概率
P_W_M = norm.pdf(130,mean_MW,std_MW)     #女性体重130磅的概率
#print(P_H_F,P_W_F,P_F_F,P_H_F*P_W_F*P_F_F*P_F,'\n')
#print(P_H_M,P_W_M,P_F_M,P_H_M*P_W_M*P_F_M*P_M,'\n')

#print(P_H_F,P_W_F,P_F_F,P_H_F*P_W_F*P_F_F*P_F,'\n')
#print(P_H_M,P_W_M,P_F_M,P_H_M*P_W_M*P_F_M*P_M,'\n')
#P_F_HWF = P_H_F*P_W_F*P_F_F*P_F/((P_H_F+P_H_M)*(P_W_F+P_W_M)*(P_F_F+P_F_M))
#P_M_HWF = P_H_M*P_W_M*P_F_M*P_M/((P_H_F+P_H_M)*(P_W_F+P_W_M)*(P_F_F+P_F_M))
P_F_HWF = P_H_F*P_W_F*P_F_F*P_F
P_M_HWF = P_H_M*P_W_M*P_F_M*P_M
print('女性概率',P_F_HWF)
print('男性概率',P_M_HWF)












