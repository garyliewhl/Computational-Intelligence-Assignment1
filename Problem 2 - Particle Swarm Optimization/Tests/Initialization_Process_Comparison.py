import random
import matplotlib.pyplot as plt

def normal_initialization(population_size, val_range):
    population = []
    for i in range(population_size):
        population.append(random.randint(val_range[0], val_range[1]))
    return population

def special_initialization(population_size, val_range, segments):
    population = []
    pop_segment = population_size//segments
    range_val = val_range[1]//segments
    min_val = 0
    max_val=range_val
    for segment in range(segments):
        for num in range(pop_segment):
            population.append(random.randint(min_val, max_val))

        min_val+=range_val
        max_val+=range_val
    return population

def main():
    population_size = 50
    val_range = [0, 1000000]

    random.seed(1000)
    population1 = normal_initialization(population_size, val_range)
    random.seed(1000)
    population2 = special_initialization(population_size, val_range, 5)
    random.seed(1000)
    population3 = special_initialization(population_size, val_range, 10)

    x1 = [0 for i in range(population_size)]
    x2 = [1 for i in range(population_size)]
    x3 = [2 for i in range(population_size)]
    
    plt.scatter(population1, x1)
    plt.scatter(population2, x2)
    plt.scatter(population3, x3)
    plt.ylabel("Initialization Process")
    plt.xlabel("Position Value")
    plt.title("Comparing Initialization Process")
    plt.savefig("Comparing Initialization Process",bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    main()