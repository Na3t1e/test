import numpy as np
# пропишем функцию потерь
def objective(w1, w2):
    return w1 ** 2 + w2 ** 2


# а также производную по первой
def partial_1(w1):
    return 2.0 * w1


# и второй переменной
def partial_2(w2):
    return 2.0 * w2

if __name__ == '__main__':
    # пропишем изначальные веса
    w1, w2 = 3, 4

    # количество итераций
    iter = 100

    # и скорость обучения
    learning_rate = 0.05

    w1_list, w2_list, l_list = [], [], []

    # в цикле с заданным количеством итераций
    for i in range(iter):
        # будем добавлять текущие веса в соответствующие списки
        w1_list.append(w1)
        w2_list.append(w2)

        # и рассчитывать и добавлять в список текущий уровень ошибки
        l_list.append(objective(w1, w2))

        # также рассчитаем значение частных производных при текущих весах
        par_1 = partial_1(w1)
        par_2 = partial_2(w2)

        # будем обновлять веса в направлении,
        # обратном направлению градиента, умноженному на скорость обучения
        w1 = w1 - learning_rate * par_1
        w2 = w2 - learning_rate * par_2

    # выведем итоговые веса модели и значение функции потерь
    w1, w2, objective(w1, w2)

    w1 = np.linspace(-5, 5, 1000)
    w2 = np.linspace(-5, 5, 1000)

    w1, w2 = np.meshgrid(w1, w2)

    f = w1 ** 2 + w2 ** 2
    print(f)


