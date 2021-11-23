import dcor
import numpy as np


def main():
    x = np.array([[1, 1], [2, 2]])
    y = np.array([[1, 3], [2, 4]])
    print(dcor.independence.distance_correlation_t_test(x,y).p_value)


if __name__ == '__main__':
    main()

