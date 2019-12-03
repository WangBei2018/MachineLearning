# P_A 小孩说谎的概率
P_B = 0.1379        # P_B 小孩可信的概率
P_B1 = 1 - P_B      # P_B 小孩不可信的概率
P_A_B = 0.1         # P_A_B 可信小孩说谎的概率
P_A1_B = 0.9        # P_A1_B 可信小孩不说谎的概率
P_A_B1 = 0.5        # P_A_B1 不可信小孩说谎概率
P_A1_B1 = 0.5       # P_A1_B1 不可信小孩不说谎概率
i=0
while(P_B < 0.8):
    P_B_A1 = P_A1_B * P_B/(P_A1_B*P_B+P_A1_B1*P_B1)
    P_B = P_B_A1
    P_B1 = 1 - P_B
    print(P_B)