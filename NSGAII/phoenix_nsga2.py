def phoenix_nsga2(ptype,num,freq,delta,rho,cond,r,k,maxeval, npop,noff,crosstype,probcr):
    # from NSGAII_copy.nsga2.problem import Problem
    # from NSGAII_copy.nsga2.individual import Individual
    # from NSGAII_copy.nsga2.evolution import Evolution
    # from NSGAII_copy.nsga2.readttp import readttp, dynp, findk, np, pd, re
    from nsga2.problem import Problem
    from nsga2.individual import Individual
    from nsga2.evolution import Evolution
    from nsga2.readttp import readttp, dynp, findk, np, pd, re
    # import matplotlib.pyplot as plt
    import math

    import pickle
    import time

    ######
    ptype = int(ptype)
    freq = int(freq)
    delta = int(delta)
    rho = float(rho)
    cond = int(cond)
    r = int(r)
    k = int(k)
    nPop = int(npop)
    nOff = int(noff)
    crosstype = int(crosstype)
    prob_crossover = float(probcr)
    number_of_variables = int(num)
    million = int(maxeval)

    if ptype == 1:
        ProblemName = 'eil101_n' + str(number_of_variables) + '_uncorr_01.ttp'
        Ptype = 'uncorr'
    elif ptype == 2:
        ProblemName = 'eil101_n' + str(number_of_variables) + '_bounded-strongly-corr_01.ttp'
        Ptype = 'bsc'
    elif ptype == 3:
        ProblemName = 'eil101_n' + str(number_of_variables) + '_uncorr-similar-weights_01.ttp'
        Ptype = 'usw'

    if crosstype == 1:
        Cross_str = 'UX'
    elif crosstype == 2:
        Cross_str = 'SPX_0'
    elif crosstype == 3:
        Cross_str = 'SPX_1'


    FinalErrList = []
    # FinalErrStr = 'MyTxt/NSGA2_' + Ptype + '_' + str(num) + '_' + str(cond) + '_' + str(r) + '_' + str(freq) + '_' + str(
    #     delta) + '_' + str(rho) + '_' + str(nPop) + '_' + Cross_str + '_' + str(k + 1) + '.txt'

    FinalErrStr = 'MyTxt/NSGA2_' + Ptype + '_' + str(num) + '_' + str(cond) + '_' + str(r) + '_' + str(freq) + '_' + str(
         delta) + '_' + str(rho) + '_' + str(k + 1) + '.txt'

    print(FinalErrStr)

    (df, InitialCapacity) = readttp(ProblemName)
    w = np.array(df.loc[:]['weight']).astype('int64')  # weight in the instance
    p = np.array(df.loc[:]['profit']).astype('int64')  # profit in the instance

    maxp = max(p)  # maximum profit in the instances
    maxweight = sum(w)  # total nominal weight possible
    # # Dynamic Table
    m = np.zeros([number_of_variables + 1, sum(w) + 1])
    DynamicTable = dynp(w, p, m, m.shape[1] - 1)
    # # shift in inital capacity
    woriginal = w - 100
    NewCapacity = findk(woriginal, InitialCapacity)
    ### Magnitude of changes:
    DistStr = 'Dist/uniform_dist_' + str(r) + '.csv'
    CapacityTable = pd.read_csv(DistStr, header=None)
    Freq = int(million) / freq
    #####
    problem = Problem(num_of_variables=number_of_variables,
                      new_capacity=NewCapacity,
                      w=w,
                      p=p,
                      maxp=maxp,
                      delta=delta,
                      rho=rho,
                      cond=cond,
                      r=r,
                      pstar=int(DynamicTable[-1, NewCapacity]))
    ##########
    global_best = Individual(length=number_of_variables)
    max_iterations=1e4
    initial_alg = Evolution(problem,
                            global_best,
                            global_error=1e9,
                            population=[],
                            num_of_iterations=int(max_iterations),
                            population_size=nPop,
                            offspring_population_size=nOff,
                            prob_mutation=(1 / number_of_variables),
                            prob_crossover=prob_crossover,
                            finalerr=[])
    t = time.time()
    #
    print('Initial phase is started')
    solutions = initial_alg.evolve()
    #

    elapsed = time.time() - t

    print('Initial Phase is finished with computing time of: {}'.format(elapsed))

    max_iterations = int(freq)

    for f in range(int(Freq)):
        NewCapacity += CapacityTable.iloc[f, k]
        if NewCapacity >= maxweight:
            NewCapacity = maxweight
        elif NewCapacity <= 0:
            NewCapacity = 0

        problem = Problem(num_of_variables=number_of_variables,
                          new_capacity=NewCapacity,
                          w=w,
                          p=p,
                          maxp=maxp,
                          delta=delta,
                          rho=rho,
                          cond=cond,
                          r=r,
                          pstar=int(DynamicTable[-1, NewCapacity]))

        algorithm = Evolution(problem,
                                global_best,
                                global_error=1e9,
                                population=solutions,
                                num_of_iterations=max_iterations,
                                population_size=nPop,
                                offspring_population_size=nOff,
                                prob_mutation=(1 / number_of_variables),
                                prob_crossover=1,
                                finalerr=[])
        solutions = algorithm.evolve()
        FinalErrList.extend(algorithm.finalerr)

        if np.mod(f,50)==0:
            print('Capacity: {}, Freq: {}, FinalErr: {}'.format(NewCapacity,f,algorithm.finalerr[-1]))

    FinalErr = np.sum(FinalErrList) / million

    with open(FinalErrStr, 'w') as of:
        of.write(str(FinalErr))
        of.write('\n')

    print('Output is done')
