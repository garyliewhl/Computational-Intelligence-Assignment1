import random, copy, time, numpy as np
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

""" Scuffed up code lmao
def validate_combination(seat, elem):
    inx = seat.index(elem)
    for i in range(1,num_people+1):
        if(seat.count(i) == 0):
            seat[inx] = i
            
        
def valid_combination(seat):
    for elem in seat:
        if seat.count(elem) > 1:
            validate_combination(seat, elem)
    return True 

def crossover(arr1,arr2):
    #Crossover/Breeding of the two parents
    cpoint1 = random.randint(1, num_people) #Crossover Point 1
    cpoint2 = random.randint(1, num_people - 1) #Crossover Point 2
    if cpoint2 >= cpoint1:
        cpoint2 += 1
    else:  # Swap the two cx points
        cpoint1, cpoint2 = cpoint2, cpoint1

    arr1[cpoint1:cpoint2], arr2[cpoint1:cpoint2] = arr2[cpoint1:cpoint2], arr1[cpoint1:cpoint2]

    return arr1,arr2
 """
 
#CONSTANTS

people = [
            [0, 20, 20, 30, 25],
            [20, 0, 50, 20, 5],
            [10, 10, 0, 100, 10],
            [50, 5, 10, 0, -5],
            [-50, -30, -100, -10,0]
        ]

num_people = 5
pop_size = 25
population = gen_pop(pop_size)
happiness_list = []
tournament = []

def clear_variables():
    global population
    population = gen_pop(pop_size)
    happiness_list.clear()
    tournament.clear()
    

#Generates the newer generation by selecting two parents and breeding them
def reproduce_offspring(population):
    
    # We will make a new list that stores the fitness/happiness of each chromosome
    fitness_list = []
    
    for chromosome in population:
        indiv_fitness = total_happiness(chromosome)
        fitness_list.append(indiv_fitness)

    average_fitness = sum(fitness_list)/len(fitness_list)
    remove = []
    remainders = []
    
    for i in range(0, len(population)):
        if fitness_list[i] < average_fitness:
            remove.append(i)
            fitness_list[i] = -1000

    for i in range (len(population),0):
        if fitness_list[i] == -1000:
            fitness_list.pop(i)

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
        next_gen.append(child)

    return next_gen
    
""" Part of the scuffed code    
    #Parents are selected from the remaining populace
    for i in range(number_removed):
        c_flag = False #initialising flag 
        counter = 0 #Child flag
        
        while(c_flag is False):
            # Pick a parent from the remainders
            parent1 = random.choice(next_gen)
            parent2 = random.choice(next_gen)
            
            child1,child2 = crossover(parent1,parent2)
            
            mutate(child1)
            mutate(child2)
            
            if(valid_combination(child1)):
                next_gen.append(child1)
                counter += 1
            if(valid_combination(child2)):
                next_gen.append(child2)
                counter+=1

            if (counter>0):
                c_flag = True
                 """

        

#Mutation function
def mutate(generation):
    mutation_prob = 0.002
    #Each member of the population has a chance of being mutated
    for i in range(len(generation)):
        if random.random() < mutation_prob:
            #Two seats are swapped at random
            a, b = random.sample(range(num_people), 2)
            population[i][b], population[i][a] = population[i][a], population[i][b]


#Calls fitness function at every iteration
def progress(generation):
    population_fitness = [total_happiness(chromosome) for chromosome in generation]
    max_location = population_fitness.index(max(population_fitness))
    tournament.append(generation[max_location])
    happiness_list.append(population_fitness[max_location])

#Replaces numbers with accurate labels
def seat_labels(seats):
    j = 0
    temp = copy.deepcopy(seats)
    
    for i in seats:
        if(i==1):
            temp[j] = "A"
        elif(i==2):
            temp[j] = "B"
        elif(i==3):
            temp[j] = "C"
        elif(i==4):
            temp[j] = "D"
        elif(i==5):
            temp[j] = "E"
            
        j = j+1            

    return temp

def main():
    clear_variables()
    global population
    number_of_generations = 1000
    gen_count = number_of_generations-1
    convergence_index = 0
    convergence_count = 0

    while number_of_generations:
        next_gen = reproduce_offspring(population)
        mutate(next_gen)
        progress(next_gen)
        population = next_gen
        happiness_list = [total_happiness(i) for i in tournament]

        #Used a counter for the number of elements in the final tournament
        if(number_of_generations<gen_count):
            #Used to compare each recurring element in the tournament with each other
            if(total_happiness(tournament[convergence_index]) < total_happiness(tournament[convergence_index+1])):
                convergence_count = 0 #If the value wavers, the convergence counter resets
            else:
                convergence_count += 1 #If the value does not change the convergence adds up
                
            convergence_index+=1

        # The termination conditions are the final iteration (1000 generations) or 
        # if the convergence count (Number of recurring elements) passes roughly 30%
        if (number_of_generations == 1 or convergence_count > (gen_count/3)):
            print("Tournament")
            #Prints and calculates the fitness/happiness of the best members of each generation
            for i in range (len(tournament)):
                happiness_list[i] = total_happiness(tournament[i])
                print("Best of Generation ", i+1)
                print(seat_labels(tournament[i])," | Happiness: ", happiness_list[i])
                
            champion_location = happiness_list.index(max(happiness_list))
            print("The Most Optimised Arrangment is : ", seat_labels(tournament[champion_location]), ", with a happiness score of: ", happiness_list[champion_location])
            break

        number_of_generations -= 1


if __name__ == '__main__':
    plt.figure(figsize=(20,10))
    plt.yticks(np.arange(-30, 250, 5))
    plt.xticks(np.arange(0, 1000, 50))
    plt.title("Happiness Scale")    
    plt.xlabel("Generation Number")
    plt.ylabel("Total Happiness")
    for i in range(10):
        main()
        plt.plot(happiness_list, label="Iteration {}".format(i+1))
        
    plt.legend(loc="lower right")
    plt.savefig("Multiple Run Comparison",bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(20,10))
    plt.xlabel("Population")
    plt.ylabel("Computiational Speed")
    computation_speed = []
    for i in range(1,1000):
        pop_size = i
        start = time.time()
        main()
        end = time.time()
        time_elapsed = end-start
        print("Time Taken: ", time_elapsed)
        computation_speed.append(time_elapsed)

    plt.plot(computation_speed)
    plt.savefig("Computational Time with Population",bbox_inches="tight")
    plt.show()

