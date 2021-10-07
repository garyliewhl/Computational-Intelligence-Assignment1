import random, time, numpy as np
import matplotlib.pyplot as plt

# Initialise and Generate Population
def gen_pop(pop_size):
    start_seating = [1,2,3,4,5]
    population = []

    #Population is filled with various permutations of seating
    for i in range(pop_size):
        random.shuffle(start_seating)
        population.append(list(start_seating))

    return population

# Fitness Function
def calc_happiness(seats):
    happiness_matrix = [
                [0, 20, 20, 30, 25],
                [20, 0, 50, 20, 5],
                [10, 10, 0, 100, 10],
                [50, 5, 10, 0, -5],
                [-50, -30, -100, -10,0]
            ]

    happiness = 0

    # Iterating seats from left to right
    for k in range(len(seats) - 1):
        # Seat indices for happiness_table
        i = seats[k] - 1
        j = seats[k + 1] - 1

        # Adding each individual pair
        pairs = happiness_matrix[i][j] + happiness_matrix[j][i]
        happiness += pairs

    return happiness

# Asexual reproduction 
def asexual_crossover(parent):
    # Get a cut-off point in the middle of the parent's gene
    cut_point = int(len(parent)/2)

    # Shuffle the gene segment after the cut-off point
    gene_segment = parent[cut_point:]
    random.shuffle(gene_segment)

    # Combine gene segments together
    child = parent[:cut_point] + gene_segment

    return child

# Generates the next generation by filtering with fitness and performing Asexual crossover to replenish lost population
def generate_next_generation(population):
    
    # Calculate average fitness
    fitness_list = [calc_happiness(chromosome) for chromosome in population]
    #print(fitness_list)
    average_fitness = sum(fitness_list)/len(fitness_list)

    # Remove any chromosome below average fitness
    remaining_pop = [population[i] for i in range(len(population)) if fitness_list[i] >= average_fitness]
    next_gen = remaining_pop.copy()
    
    # Parent is randomly selected from the remaining population and crossedover to replenish lost population
    number_removed = len(population) - len(remaining_pop)
    for i in range(number_removed):
        #print("remaining:", remaining_pop)
        # Pick a parent from the remainders
        child = asexual_crossover(random.choice(remaining_pop)) 
        next_gen.append(child)

    return next_gen        

# Mutation function
def mutate(population, mutation_prob):
    # Each member of the population has a chance of being mutated
    for chromo_index in range(len(population)):
        # Roll mutation probability
        if random.random() <= mutation_prob:
            # Swaps two random seat if chosen for mutation
            a, b = random.sample(range(len(population[chromo_index])), 2)
            population[chromo_index][b], population[chromo_index][a] = population[chromo_index][a], population[chromo_index][b]

    return population

# Selects the best chromosome from this generation
def progress(population):
    population_fitness = [calc_happiness(chromosome) for chromosome in population]
    best_index = population_fitness.index(max(population_fitness))
    return population[best_index]

#Replaces numbers with accurate labels
def seat_labels(seats):
    label_list = []
    
    for i in seats:
        if(i==1):
            label_list.append("A")
        elif(i==2):
            label_list.append("B")
        elif(i==3):
            label_list.append("C")
        elif(i==4):
            label_list.append("D")
        else:
            label_list.append("E")

    return label_list
    
def main():
    
    plt.figure(figsize=(5,5))
    computational_time = []
    for i in range (1,201):
        random.seed(10)
        # Parameters
        pop_size = i
        start = time.time()    
        happiness_list = []
        tournament_winners = []    
        max_iter = 1000
        curr_iter = 1
        convergence_count = 0
        convergence_limit = 300
        mutation_prob = 0.002

        # Step 1: Initialization
        population = gen_pop(pop_size)

        # Termination condition 1: Max iteration 
        while curr_iter <= max_iter:
            #print("Population:", population)
            # Termination condition 2: No improvement after specified iteration
            if (convergence_count > convergence_limit):
                break
            
            # Step 2: Check fitness and record best chromosome
            tournament_winners.append(progress(population))
            happiness_list.append(calc_happiness(tournament_winners[-1]))
            # print("Tournament:", tournament_winners)
            # print("Happiness:", happiness_list)
            # Step 3: Make a new generation
            population = generate_next_generation(population)

            # Step 4: Mutate 
            population = mutate(population, mutation_prob)

            # Check for tournament winner improvements
            if(curr_iter>1):
                # If improvement occurs, reset convergence_count
                if(happiness_list[-1] > happiness_list[-2]):
                    convergence_count = 0 
                else:
                    convergence_count += 1 

            curr_iter += 1

        print("Tournament")
        #Prints and calculates the fitness/happiness of the best members of each generation
        for i in range (len(tournament_winners)):
            print("Best of Generation ", i+1)
            print(seat_labels(tournament_winners[i])," | Happiness: ", happiness_list[i])
            
        champion_location = happiness_list.index(max(happiness_list))
        print("The Most Optimized Arrangement is : ", seat_labels(tournament_winners[champion_location]), ", with a happiness score of: ", happiness_list[champion_location])
        computational_time.append(time.time()-start)

    plt.plot(computational_time)
    plt.xlabel("Population")
    plt.ylabel("Computational Speed")
    plt.savefig("Computational Time vs Population - GA", bbox_inches="tight")
    plt.show()
        
if __name__ == '__main__':
    main()