import serial
ser = serial.Serial('COM11', 9600) # giao tieepa arduino với python
# Thiết lập kết nối Serial với Arduino
def move(command, speedL, speedR): #hàm vận chuyển dữ liệu từ py-ino
    # Tính toán tốc độ các bánh xe
    speed1 = int(speedL ) # toc do banh xe 1,234
    speed2 = int(speedR )
    speed3 = int(speedL )
    speed4 = int(speedR )
    # Gửi tốc độ đến Arduino qua kết nối Serial
    ser.write(command.encode())# gọi hàm gửi command
    ser.write(b',')
    ser.write(str(speed1).encode()) #hàm gửi toc do speed 1234
    ser.write(b',') # cách nhau 1 dấu phẩy
    ser.write(str(speed2).encode())
    ser.write(b',')
    ser.write(str(speed3).encode())
    ser.write(b',')
    ser.write(str(speed4).encode())
    ser.write(b'\n')
    # print(command, speed1,speed2,speed3,speed4)
def distance(): # hàm nhận khoảng cách từ arduino
    dis = ser.readline()
    dis = str(dis, "utf-8")
    dis = dis.strip('\r\n')
    Splitdis = dis.split(',')
    dis1 = float(Splitdis[0])
    dis2 = float(Splitdis[1])
    RPM1 = float(Splitdis[2])
    RPM3 = float(Splitdis[3])
    return dis1, dis2, RPM1, RPM3

# while 1:
#     n = input("Enter a number: ")
#     if int(n) == 2:
#         move('yeu_cau', 100, 100)
#         x, y = distance()
#         print("Distance 1:", x)
#          print("Distance 2:", y)
