from turtle import color
import matplotlib.pyplot as plt

list = []


def read_csv():
    f = open("/home/fmasa/catkin_ws/src/detect_white_line/image/csv/haarlike.csv", 'r')
    for row in f:
        list.append(row)


read_csv()
# print(list)

for i in range(391):
    res = list[i]
    pos = res.replace("\n", "").split(',')
    # print(pos)
    x = float(pos[0])
    y = float(pos[1])
#     print(x)
#     print(y)
#     print(type(x))
    plt.plot(x, y, marker="o", color="red")
plt.show()
