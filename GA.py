import random
import matplotlib.pyplot as plt

#Initialise and Generate Population
def gen_pop(pop_size):
    start_seating = [1,2,3,4,5]
    population = []

    #Population is filled with various permutations of seating
    for i in range(pop_size):
        random.shuffle(start_seating)
        population.append(list(start_seating))

    return population

#Fitness Function
def total_happiness(seats):
    happiness = 0

    for k in range(num_people - 1):
        #These are our indices
        i = seats[k] - 1
        j = seats[k + 1] - 1

        #Adding each individual pair
        pairs = people[i][j] + people[j][i]
        happiness = happiness + pairs

    return(happiness)

#CONSTANTS

people = [
            [0, 20, 20, 30, 25],
            [20, 0, 50, 20, 5],
            [10, 10, 0, 100, 10],
            [50, 5, 10, 0, -5],
            [-50, -30, -100, -10,0]
        ]

num_people = 5
pop_size = 10
population = gen_pop(pop_size)
happiness_list = []
tournament = []

#Generates the newer generation by selecting two parents and breeding them
def reproduce_offspring(population):
    
    # We will make a new list that stores the fitness/happiness of each chromosome
    fitness_list = []
    
    for chromosome in population:
        indiv_fitness = total_happiness(chromosome)
        fitness_list.append(indiv_fitness)

    average_fitness = sum(fitness_list)/len(fitness_list)
    remove = []

    for i in range(0, len(population)):
        if fitness_list[i] < average_fitness:
            remove.append(i)

    remainders = [i for j, i in enumerate(population) if j not in remove]       
    
    number_removed = len(population) - len(remainders)

    next_gen = remainders
    #Parents are selected from the remaining populace
    for i in range(number_removed):
        # Pick a parent from the remainders
        parent = random.choice(remainders)
        #Making use of parents from the same sample caused an issue of duplicated seats eg: [A,B,C,D,A]
        #Thus parent 2 is created by modifying parent 1 
        parent2 = parent[int(num_people / 2):num_people]
        random.shuffle(parent2)
        #Crossover (Child is bred using half of parent 1 and half of parent 2)
        child = parent[0:int(num_people / 2)] + parent2

        #New Generation
        next_gen.append(child)

    return next_gen

#Mutation function
def mutate(generation):
    mutation_prob = 0.0015
    #Each member of the population has a chance of being mutated
    for i in range(len(generation)):
        if random.random() < mutation_prob:
            #Two seats are swapped at random
            a, b = random.sample(range(num_people), 2)
            population[i][b], population[i][a] = population[i][a], population[i][b]

#Calls fitness function at every iteration
def progress(generation):
    population_fitness = [total_happiness(chromosome) for chromosome in generation]
    happiness_list.append(max(population_fitness))
    max_location = population_fitness.index(max(population_fitness))
    tournament.append(population[max_location])

#Replaces numbers with accurate labels
def seat_labels(champion):
    j = 0
    for i in champion:
        if(i==1):
            champion[j] = "A"
        elif(i==2):
            champion[j] = "B"
        elif(i==3):
            champion[j] = "C"
        elif(i==4):
            champion[j] = "D"
        elif(i==5):
            champion[j] = "E"
        j = j+1
        
    return champion

def main():
    number_of_generations = 1000
    global population
    
    while number_of_generations:
        next_gen = reproduce_offspring(population)
        progress(next_gen)
        mutate(next_gen)
        population = next_gen
        champion_location = happiness_list.index(max(happiness_list))

        if number_of_generations == 1:
            print("Tournament")
            for i in range (len(tournament)):
                happiness_list[i] = total_happiness(tournament[i])
                print("Best of Generation ", i+1)
                print(tournament[i]," | Happiness: ", happiness_list[i])
                
            champion_location = happiness_list.index(max(happiness_list))
            print("The Most Optimised Arrangment is : ", seat_labels(tournament[champion_location]), ", with a happiness score of: ", happiness_list[champion_location])

        number_of_generations -= 1


    plt.plot(happiness_list)
    plt.title('Happiness Scale')    
    plt.xlabel('Generation Number')
    plt.ylabel("Total Happiness")
    plt.show()

if __name__ == '__main__':
    main()