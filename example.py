current = 190
previous = 100
current_A = 190
previous_A = 100
reward = 100
while True:
    tmp = current
    previous = tmp
    current = current * 0.9 + reward
    current_A = reward + current*0.9
    reward -= 1
    print(current, previous, current_A)
    if current - previous == 0 or current - previous < 0:
        break
print(current)
print(current_A)