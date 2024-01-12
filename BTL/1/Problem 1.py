import numpy as np
import cvxpy as cp

# Khởi tạo các tham số
n = 8  # Số lượng sản phẩm
S = 2  # Số lượng kịch bản
p_s = 1/2  # Mật độ
m = 5  # Số lượng phần cần đặt trước khi sản xuất

# Tạo ma trận A với các giá trị nguyên dương ngẫu nhiên từ 1 đến 10
A = np.random.randint(1, 11, size=(n, m))
# Tạo vector nhu cầu ngẫu nhiên D theo phân phối nhị thức Bin(10, 1/2)
D = np.random.binomial(10, 1/2, (S, n))
# Tạo vector b với các giá trị ngẫu nhiên trong khoảng từ 0 đến 50
b = np.random.uniform(0, 50, size=m)
# Tạo vector l với các giá trị ngẫu nhiên trong khoảng từ 60 đến 1200
l = np.random.uniform(100, 2400, size=n)
# Tạo vector q với các giá trị ngẫu nhiên trong khoảng từ 1000 đến 10000
q = np.random.uniform(1000, 10000, size=n)
# Tạo vector s với các giá trị ngẫu nhiên nhỏ hơn b
s = np.array([np.random.uniform(0, b_j, size=1)[0] for b_j in b])


print("b = ", b) 
print("l = ", l)
print("q = ", q)
print("s = ", s)
print("D =\n", D)
print("A =\n", A)

# Khởi tạo biến
x = cp.Variable(m, integer=True) #số lượng bộ phận cần đặt trước khi sản xuất.
y = cp.Variable((S, m), integer=True) #số lượng linh kiện của từng bộ phận.
z = cp.Variable((S, n), integer=True) #số lượng bộ phận sau khi đã trừ đi số lượng linh kiện đã được sử dụng để sản xuất sản phẩm.

# Tính c
c = l - q

# Xây dựng hàm mục tiêu
objective = cp.Minimize(b.T @ x + p_s * (cp.sum(c.T @ z[1, :]) - cp.sum(s.T @ y[1, :])) + p_s * (cp.sum(c.T @ z[0, :]) - cp.sum(s.T @ y[0, :])))

# Xây dựng ràng buộc
constraints = [y[s, :] == x - A.T @ z[s, :] for s in range(S)]
constraints += [z[s, :] >= 0 for s in range(S)]
constraints += [z[s, :] <= D[s, :] for s in range(S)]
constraints += [y[s, :] >= 0 for s in range(S)]

# Giải quyết vấn đề tối ưu hóa
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.CBC)

# In kết quả
print("\nGiá trị tối ưu:", prob.value)
print("Giải pháp x:", x.value)
print("Giải pháp y:\n", y.value)
print("Giải pháp z:\n", z.value)
