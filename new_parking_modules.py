import numpy as np
import datetime

cmd = 2
zks = 10
parking = np.zeros((cmd, zks))

parking_log = {}
discount = {}
fee_per_min = 50

def management() :
    global fee_per_min, parking, cmd, zks, discount, parking_log, discount
    while True :
        admin_request = input(f"""1. 주차 로그 확인
2. 정기주차 등록
3. 요금 관리
4. 주차장 관리 (모든 기록이 초기화됩니다.)
5. 나가기""")
        
        match admin_request :
            case "1" :
                print(parking_log)
            case "2" :
                n_car = input("새로 등록할 차량 번호 : ")
                n_dis = int(input("할인율 : "))
                discount[n_car] = n_dis
                print("할인 차량 목록 :", discount)
            case "3" :
                fee_per_min = int(input("분당 요금 : "))
            case "4" :
                cmd = int(input("새로운 주차장의 층 : "))
                zks = int(input("층별 주차 칸 수 : "))
                parking = np.zeros((cmd, zks))
                parking_log = {}
                discount = {}
                display()
            case "5" :
                break

def display() :
    global parking, cmd, zks
    la_list = []
    for a in range(65, zks+65) :
        la_list.append(chr(a))
    label = "        "+"    ".join(la_list)+"    여유 공간"
    park2 = np.where(parking == 0, "[  ]", "[🚗]")
    print(label)
    floor = cmd
    for idx, pa in enumerate(reversed(park2)) :
        empty = np.count_nonzero(parking[::-1][idx]==0)
        print(f"{floor}층 | ", end='')
        print(*pa, end='      ')
        print(f"{empty}")
        floor-=1

def car_in(car_num) :
    global parking_log, parking
    while True :    
        time_in = datetime.datetime.now()
        tar_floor = int(input("원하는 층 : "))     
        tar_loc = input("원하는 자리 : ").upper()
        idx_fl = tar_floor-1
        idx_loc = ord(tar_loc)-65
        if parking[idx_fl][idx_loc] == 0 :
            break
        else :
            print("이미 주차된 자리입니다.")
    parking[idx_fl][idx_loc] = 1
    parking_log[car_num] = [(idx_fl, idx_loc), time_in, 0, 1]

def car_out(car_num) :
    global parking_log, parking, fee_per_min
    time_out = datetime.datetime.now()
    parking_log[car_num][2] = time_out
    time = time_out - parking_log[car_num][1]
    print(f"""========={car_num} 출차 정보=========
주차 위치 | {parking_log[car_num][0][0]+1}층 {chr(parking_log[car_num][0][1]+65)}
주차 시간 | {time}
주차 요금 |""", end=' ')
    if car_num in discount.keys() :
        cost = fee_per_min*np.ceil(time.seconds//60/10)*(100-discount[car_num])/10
        print(f"{int(cost)}원 ({discount[car_num]}% 할인)")
    else :
        cost = fee_per_min*np.ceil(time.seconds//60/10)*10
        print(f"{int(cost)}원")
    parking[parking_log[car_num][0][0]][parking_log[car_num][0][1]] = 0
    parking_log[car_num][3] = 0

while True :
    display()
    car_num = input("차량 번호를 입력해주세요 : ")
    if car_num == "exit" :
        break
    elif car_num == "admin" :
        management()
    elif car_num in parking_log :
        car_out(car_num)
    elif car_num not in parking_log:
        car_in(car_num)