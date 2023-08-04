# Ryan Nathaniel B. Resoles
# CMSC 170 ST-1L Final Project 
# Aircraft Landing Scheduling Problem (ALSP) using Genetic Algorithm

import random
import numpy as np
import math
import sys
import time

INITIAL_POP = 10
MAXPOP = 1000
PAIR_COUNT = 4

# plane object
class Plane:
    def __init__(self, no, arrival, earliest, target, latest, ep, lp):
        self.plane_no = no
        self.arrival = arrival
        self.earliest = earliest
        self.target = target
        self.latest = latest
        self.early_penalty = ep
        self.late_penalty = lp

class Schedule:
    def __init__(self, sched, sep):
        self.sched = sched
        self.separation = sep
        self.placeholder = self.calculate_fitness()
        self.fitness = self.placeholder[0]
        self.chance = 0
        self.finish = self.placeholder[1]
        self.valid = self.placeholder[2]
    
    # calculates fitness based on penalties (computes total penalty)
    # penalty is inversely proportional to fitness
    def calculate_fitness(self):
        penalty = 0
        time = self.sched[0].target
        valid = 1

        for i in range(1, len(self.sched)):
            time += self.separation[i]
            temp_penalty = 0

            # invalid scheduling if current time > latest time
            if self.sched[i].latest < time:
                valid = 0
                break
            
            # late schedule
            if self.sched[i].target < time and self.sched[i].latest >= time:
                penalty += ((time - self.sched[i].target)*self.sched[i].late_penalty)
                continue

            # early schedule
            if self.sched[i].target < time: 
                if self.sched[i].earliest < time:
                    penalty += ((self.sched[i].target - self.sched[i].earliest)*self.sched[i].early_penalty)
                else:
                    penalty += ((self.sched[i].target - time)*self.sched[i].early_penalty)
                time = self.sched[i].target
        

        return [penalty, time, valid]

    def mutate(self, newSched):
        self.sched = newSched
        self.placeholder = self.calculate_fitness()
        self.fitness = self.placeholder[0]
        self.finish = self.placeholder[1]
        self.valid = self.placeholder[2]

def readFile(filename):
    fileReader = open(filename, "r")
    adj_matrix = []
    plane_vals = []

    nplanes = 0
    ctr = 0
    ctr2= 0

    flag = 0
    line_count1 = 0
    line_count2 = 0

    adder = []

    for i in fileReader:
        if i=='\n': continue
        i = i[1:].split(" ")
        if i[-1] == "\n": i=i[:-1]
        else: i[-1] = i[-1][:-1]

        if ctr2==0:
            nplanes = int(i[0])
            ctr2+=1
            continue 
        
        if flag==0:
            line = [float(x) for x in i]
            line_count1 += len(line)
            plane_vals.append(line)
            ctr+=1
        
        if flag==1:
            line = [int(x) for x in i]
            line_count2 += len(line)
            adder += line
            ctr+=1

        if line_count1==6:
            flag=1
            line_count1=0

        if line_count2==nplanes:
            flag=0
            line_count2=0
            adj_matrix.append(adder)
            adder = []

    fileReader.close()

    return [nplanes, plane_vals, adj_matrix]

# computes chance of getting selected for roulette wheel selection
def compute_chance(sched_list):
    chance = sum([x.fitness for x in sched_list])
    total_chance = 0
    for i in sched_list:
        total_chance += i.fitness/chance
        i.chance = total_chance

    return chance

def mutation(schedule):
    x = random.randint(0,len(schedule.sched)-2)
    y = random.randint(0,len(schedule.sched)-2)

    if x>y:
        x, y = y, x

    new_path = schedule.sched[:x] + [schedule.sched[y]] + schedule.sched[x+1:y] + [schedule.sched[x]] + schedule.sched[y+1:]
    schedule.mutate(new_path)

def selection(sched_list):
    pairs = []

    for i in range(PAIR_COUNT):
        temp = np.random.uniform(0,1,2)
            
        ind1, ind2 = 0, 0
        for j in range(1,len(sched_list)):
            if temp[0] > sched_list[j].chance:
                ind1 = j
            if temp[1] > sched_list[j].chance:
                ind2 = j    
        pairs.append((ind1,ind2))    

    return pairs

# crossover using order crossover, swaps every bit
# except bits appearing in selected portion
def order_crossover(mating_population, sched_list, plane_count):
    offsprings_holder = []
    os = []

    temp = random.sample(range(1, plane_count), 2)
    if temp[0] > temp[1]:
        t = temp[0]
        temp[0] = temp[1]
        temp[1] = t

    for i in mating_population:
        os = ([0]*temp[0]) + [*sched_list[i[0]].sched[temp[0]:temp[1]]] + ([0]*(plane_count-temp[1]))

        for j in range(len(os)):
            for k in sched_list[i[1]].sched:
                if os[j]==0 and k not in os:
                    os[j] = k
                    break

        offsprings_holder.append(os)

    return offsprings_holder             

def get_separation(sched, adj_matrix, plane_count):
    sep_indiv = []
    prev = sched[0].plane_no
    for i in range(0,plane_count):
        curr = sched[i].plane_no
        sep_indiv.append(adj_matrix[prev][curr])
        prev = curr
    
    return sep_indiv

def addOffsprings(offsprings, sched_list, adj_matrix, plane_count):
    for i in offsprings:
        if len(sched_list)<MAXPOP: sched_list.append(Schedule(i, get_separation(i, adj_matrix, plane_count)))
        else: sched_list[sched_list.index(max(sched_list, key=lambda x:x.fitness))] = Schedule(i, get_separation(i, adj_matrix, plane_count))



###########################################################
######################## MAIN CODE ########################          
###########################################################
if __name__=="__main__":
    plane_list = []
    sched_list = []
    content_holder = readFile('ALSP01.txt')
    plane_count = int(content_holder[0])
    plane_vals = content_holder[1]
    adj_matrix = content_holder[2]

    for i in range(plane_count):
        plane_list.append(Plane((i), plane_vals[i][0], plane_vals[i][1], plane_vals[i][2], plane_vals[i][3], plane_vals[i][4], plane_vals[i][5]))      

    while True:
        num_iter = int(input("Number of iterations (should be above or equal to 3): "))
        if num_iter > 2: break

    total_fitness=start=generation = 0
    duration = 30
    for i in range(num_iter-1):
        # increments pair count every 5 generations
        if i%5==0: PAIR_COUNT+=1
        
        # creates initial population
        if start==0:
            start_time = time.time()

            while len(sched_list)==0:
                elapsed_time = time.time() - start_time
                if elapsed_time >= duration:
                    print("Max time reached. Valid solution not found.")
                    sys.exit(1)
                for i in range(INITIAL_POP):
                    order = random.sample(range(0,plane_count), plane_count)
                    #print(order)
                    sched_indiv = []
                    sep_indiv = []
                    prev = order[0]
                    for j in range(0,plane_count):
                        curr = next(x for x in plane_list if x.plane_no==order[j])
                        sched_indiv.append(curr)
                        sep_indiv.append(adj_matrix[prev][curr.plane_no])
                        prev = curr.plane_no
                    
                    sched_list.append(Schedule(sched_indiv, sep_indiv))                

                    # mutation starts [TO BE IMPLEMENTED]
                    for i in sched_list:
                        mut_rate = random.randint(1,100)
                        if mut_rate==1:
                            mutation(i)

                sched_list = [x for x in sched_list if x.valid==1]
                compute_chance(sched_list)
                
            #print([x.valid for x in sched_list])
            
        
        total_chance = compute_chance(sched_list)

        if start==0:
            ind = min(sched_list, key=lambda x:x.fitness)
            print(f'GEN#{generation+1}: path:{[x.plane_no for x in ind.sched]} | fitness:{ind.fitness} | validity:{ind.valid}')
            generation+=1
            start+=1     

        # selection/crossover begins
        mating_population = selection(sched_list)
        offsprings = order_crossover(mating_population, sched_list, plane_count)
        addOffsprings(offsprings, sched_list, adj_matrix, plane_count)
        sched_list = [x for x in sched_list if x.valid==1]
        compute_chance(sched_list)
        generation += 1
        
        ind = min(sched_list, key=lambda x:x.fitness)
        print(f'GEN#{generation} | Path:{[x.plane_no for x in ind.sched]} | Penalty:{ind.fitness} | Time Taken:{ind.finish} | Validity:{ind.valid}')  