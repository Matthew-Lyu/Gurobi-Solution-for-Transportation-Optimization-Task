# 导入相关的包
import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model
import pandas as pd
import numpy as np
import pandas as pd
import os

'''
————————————————导入数据-————————————————
'''

n = 6  # 舱室的数量
m = 10 # 每个舱室的层数

data_input_path = './input_cargo_capacity.xlsx'
df = pd.read_excel(data_input_path, engine='openpyxl',header = None)
df_reversed = df.iloc[::-1]
df_reversed = df_reversed.reset_index(drop=True)

# 定义各舱室各层的最大容量 (i,j)表示第i舱第j层的最大容量
# 生成哈希表
C = {}
for row in range(len(df_reversed)):
    for col in range(len(df_reversed.columns)):
        cell_value = df_reversed.iat[row, col]
        C[(col + 1, row + 1)] = cell_value

S = 3  # 装载地的数量，编号为1，2，3
D = 4  # 送达地的数量，编号为1，2，3，4 # 用1,2,3,4表示A,B,C,D

# 每个装载地能提供纸浆包数量
P = {1:9320, 2:6480, 3:13200}

# 每个送达地需要的纸浆包数量 # Q[s, d], s表示装载地, d表示送达地
Q = {(1,1):0, (2,1):2835, (3,1):1575,
     (1,2):360, (2,2):3645, (3,2):4725,
     (1,3):6355, (2,3):0, (3,3):2490,
     (1,4):2605, (2,4):0, (3,4):4410}

# 大M
M = 1e5


'''
————————————————定义模型————————————————
'''

# 调用Model方法创建一个模型
model = Model("PaperShip")



'''
————————————————定义决策变量————————————————
'''

cargo = {} # 构造一个字典，cargo[i, j, s, d]表示第i舱第j层从s发往d的货物
has_cargo = {} # 辅助变量，用来表示第i舱第j层是否有从s发往d的货物
for i in range(1, n + 1):
    for j in range(1, m + 1):
        for s in range(1, S + 1):
            for d in range(1, D + 1):
                # 决策的是每个舱室每一层从s发往d的货物数量
                cargo[i, j, s, d]  = model.addVar(vtype=GRB.INTEGER, name=f"cargo_{i}_{j}_{s}_{d}")
                model.addConstr(cargo[i, j, s, d] >= 0)
                has_cargo[i, j, s, d] = model.addVar(vtype=GRB.BINARY, name=f"has_cargo_{i}_{j}_{s}_{d}")
                # 在整个船上，最大的存储量是630，所以cargo[i, j, s, d]/M一定是1或者一个0到1之间的小数，添加一个约束
                model.addConstr(has_cargo[i, j, s, d] >= cargo[i, j, s, d]/M)

'''
————————————————定义约束条件————————————————
'''

# 约束1
# 舱室容量限制，第i舱室第j层的货量不能超过C[i,j]
for i in range(1, n+1):
    for j in range(1, m+1):
        # 调用addConstr方法添加约束
        # 调用quicksum方法计算第i舱第j层的货物总量
        model.addConstr(gp.quicksum(cargo[i, j, s, d]*has_cargo[i,j,s,d] for s in range(1, S+1) for d in range(1, D+1)) <= C[i, j]
                        , "CapacityLimitation")

# 约束2
# 装载地供应限制，所有从装载地s装载的纸浆包数量不能超过P[s]
for s in range(1, S+1):
    model.addConstr(gp.quicksum(cargo[i, j, s, d]*has_cargo[i,j,s,d] for i in range(1, n+1) for j in range(1, m+1) for d in range(1, D+1)) <= P[s]
                     , "SupplyLimitation")

# 约束3
# 送达地需求限制，所有送达地d的需求必须满足到货量不能小于Q[s,d]
for s in range(1, S+1):
    for d in range(1, D+1):
        model.addConstr(gp.quicksum(cargo[i, j, s, d]*has_cargo[i,j,s,d] for i in range(1, n+1) for j in range(1, m+1)) == Q[s, d]
                , "DemandLimitation")

# 约束4
# 装卸限制，由于船舶是从装载地 1 出发，顺序经过装载地 1、2、3、送达地 A、B、C、D
# 在每个舱室，来自装载地s的纸浆包总在来自装载地s+1的纸浆包之下
# 在每个舱室，来自送达地d的纸浆包总在来自送达地d+1的纸浆包之上
for i in range(1, n + 1):
    for j in range(1, m + 1):
        for s in range(1, S + 1):
            has_cargo_from_s = model.addVar(vtype = GRB.BINARY)
            model.addConstrs(has_cargo_from_s >= has_cargo[i, j, s, d_]/M for d_ in range(1, D+1))
            has_cargo_from_s_plus = model.addVar(vtype = GRB.BINARY)
            model.addConstrs(has_cargo_from_s_plus >= has_cargo[i, j_, s_, d_]/M for j_ in range(1, j) for s_ in range(s+1, S+1) for d_ in range(1, D+1))
            model.addGenConstrIndicator(has_cargo_from_s, 1, has_cargo_from_s_plus == 0)
        for d in range(1, D + 1):
            has_cargo_to_d = model.addVar(vtype = GRB.BINARY)
            model.addConstrs(has_cargo_to_d >= has_cargo[i, j, s_, d]/M for s_ in range(1, S+1))
            has_cargo_to_d_plus = model.addVar(vtype = GRB.BINARY)
            model.addConstrs(has_cargo_to_d_plus >= has_cargo[i, j_, s_, d_]/M for j_ in range(1, j) for s_ in range(1, S+1) for d_ in range(1, d))
            model.addGenConstrIndicator(has_cargo_to_d, 1, has_cargo_to_d_plus == 0)

# 约束5
# 在每个装载或卸载地时，不能同时打开 2、3 舱和 4、5 舱
# 即2，3舱和4，5舱不能出现存在cargo[2, j, s_, d_]和cargo[3, j, s_, d_]同时不为0或cargo[4, j, s_, d_]和cargo[5, j, s_, d_]同时不为0的情况
for s in range(1, S+1):
    cargo_in_2_from_s = model.addVar(vtype = GRB.BINARY)
    cargo_in_3_from_s = model.addVar(vtype = GRB.BINARY)
    cargo_in_4_from_s = model.addVar(vtype = GRB.BINARY)
    cargo_in_5_from_s = model.addVar(vtype = GRB.BINARY)
    model.addConstrs(cargo_in_2_from_s >= cargo[2, j, s, d_]/M for j in range(1, m+1) for d_ in range(1, D+1))
    model.addConstrs(cargo_in_3_from_s >= cargo[3, j, s, d_]/M for j in range(1, m+1) for d_ in range(1, D+1))
    model.addConstrs(cargo_in_4_from_s >= cargo[4, j, s, d_]/M for j in range(1, m+1) for d_ in range(1, D+1))
    model.addConstrs(cargo_in_5_from_s >= cargo[5, j, s, d_]/M for j in range(1, m+1) for d_ in range(1, D+1))
    model.addConstr(cargo_in_2_from_s + cargo_in_3_from_s <= 1)
    model.addConstr(cargo_in_4_from_s + cargo_in_5_from_s <= 1)

for d in range(1, D+1):
    cargo_in_2_to_d = model.addVar(vtype = GRB.BINARY)
    cargo_in_3_to_d = model.addVar(vtype = GRB.BINARY)
    cargo_in_4_to_d = model.addVar(vtype = GRB.BINARY)
    cargo_in_5_to_d = model.addVar(vtype = GRB.BINARY)
    model.addConstrs(cargo_in_2_to_d >= cargo[2, j, s_, d]/M for j in range(1, m+1) for s_ in range(1, S+1))
    model.addConstrs(cargo_in_3_to_d >= cargo[3, j, s_, d]/M for j in range(1, m+1) for s_ in range(1, S+1))
    model.addConstrs(cargo_in_4_to_d >= cargo[4, j, s_, d]/M for j in range(1, m+1) for s_ in range(1, S+1))
    model.addConstrs(cargo_in_5_to_d >= cargo[5, j, s_, d]/M for j in range(1, m+1) for s_ in range(1, S+1))
    model.addConstr(cargo_in_2_to_d + cargo_in_3_to_d <= 1)
    model.addConstr(cargo_in_4_to_d + cargo_in_5_to_d <= 1)


# 更新模型以集成变量和约束
model.update()


'''
————————————————定义优化目标————————————————
'''

# 目标函数
# 不同装载地-送达地的纸浆包混装在同层的次数尽可能少


mix = {}
for i in range(1, n+1):
    for j in range(1, m+1):
        mix[i,j] = model.addVar(vtype = GRB.BINARY)
        model.addConstr(gp.quicksum(has_cargo[i, j, s, d] for s in range(1, S+1) for d in range(1, D+1)) >= 1)
        model.addConstr(mix[i,j] >= (gp.quicksum(has_cargo[i, j, s, d] for s in range(1, S+1) for d in range(1, D+1))-1)/M)
        model.addConstr(gp.quicksum(cargo[i, j, s, d] for s in range(1, S+1) for d in range(1, D+1)) >= C[i, j] - M * mix[i,j])

model.setObjective(gp.quicksum(mix[i,j] for i in range(1, n+1) for j in range(1, m+1)),GRB.MINIMIZE)

'''
————————————————求解模型————————————————
'''

model.Params.MIPFocus = 3

# 启动求解过程
model.optimize()


excel_dir = './output.xlsx'
model_dir = './model.lp'

# 检查求解状态
if model.status == GRB.OPTIMAL:
    print("最优解已找到.")

    # 保存模型的求解过程
    model.write(model_dir)

    for key in cargo:
        if cargo[key].X > 0:
            print(f"Cargo in compartment {key[0]}, layer {key[1]}, from source {key[2]} to destination {key[3]}: {cargo[key].X}")

    print(f"Optimal objective value: {model.ObjVal}")

    # 将运载方案写入excel
    write = np.empty((10, 6),dtype='<U100')
    for j in range(m, 0, -1):
        for i in range(1, n+1):
            for s in range(1, S+1):
                for d in range(1, D+1):
                    if cargo[i,j,s,d].X != 0:
                        write[m-j,i-1]=write[m-j,i-1]+str(f"({s}, {d}):{cargo[i,j,s,d].X}  ") 

    df = pd.DataFrame(write)
    # 将 DataFrame 写入 Excel 文件
    df.to_excel(excel_dir, index=False, header=False)

elif model.status == GRB.INFEASIBLE:
    print("模型无解，请检查约束条件.")
elif model.status == GRB.INF_OR_UNBD:
    print("模型无解或无界，请检查模型定义.")
else:
    print("优化过程未成功完成.")
