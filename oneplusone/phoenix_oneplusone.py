def phoenix_oneplusone(ptype,num,freq,delta,rho,cond,r,k,maxeval):

    from readttp import readttp, dynp, findk, np, pd, re
    from oneplusone_script import individual, oneplusone
    import copy

    ptype = int(ptype)
    freq = int(freq)
    delta = int(delta)
    rho = float(rho)
    cond = int(cond)
    r = int(r)
    k = int(k)  # SampleTime
    num = int(num)
    maxeval = int(maxeval)

    if ptype == 1:
        ProblemName = 'eil101_n'+str(num)+'_uncorr_01.ttp'
        Ptype = 'uncorr'
    elif ptype == 2:
        ProblemName = 'eil101_n'+str(num)+'_bounded-strongly-corr_01.ttp'
        Ptype = 'bsc'
    elif ptype == 3:
        ProblemName = 'eil101_n'+str(num)+'_uncorr-similar-weights_01.ttp'
        Ptype = 'usw'

    FinalErrList = []
    FinalErrStr = 'MyTxt/oneplusone_' + str(Ptype) + '_' + str(num) + '_' + str(cond) + '_' + str(r) + '_' + str(freq) + '_' + str(
        delta) + '_' + str(rho) + '_' + str(k + 1) + '.txt'

    print(FinalErrStr)

    NumberItems = int(re.findall(r'\d+', ProblemName)[1])

    (df, InitialCapacity) = readttp(ProblemName)

    w = np.array(df.loc[:]['weight']).astype('int64')  # weight in the instance
    p = np.array(df.loc[:]['profit']).astype('int64')  # profit in the instance

    million = maxeval
    maxp = max(p)  # maximum profit in the instances
    maxweight = sum(w)  # total nominal weight possible
    # # Dynamic Table
    m = np.zeros([NumberItems + 1, sum(w) + 1])
    DynamicTable = dynp(w, p, m, m.shape[1] - 1)
    # # shift in inital capacity
    woriginal = w - 100
    NewCapacity = findk(woriginal, InitialCapacity)

    # NewCapacity=InitialCapacity
    binary_string_length = NumberItems

    ### Magnitude of changes:
    DistStr = 'Dist/uniform_dist_' + str(r) + '.csv'
    CapacityTable = pd.read_csv(DistStr, header=None)
    Freq = int(million) / freq
    Iter = 1
    ## 1e4 initial stage
    solution = individual(NumberItems)
    initial_alg = oneplusone(solution, NewCapacity, w, p, maxp, delta, rho, cond, r, int(DynamicTable[-1, NewCapacity]))
    initial_alg.evaluate()
    for Iter in range(int(1e4)):
        initial_alg.mutate()
        initial_alg.calc_err()
        # if np.mod(Iter,1000)==0:
        #     print(Iter)

    solution = copy.deepcopy(initial_alg.individual)
    # print(initial_alg.finalerr[-1],Iter)

    # print('Initial Phase is finished with computing time of: ',str(algorithm.total_computing_time))
    print('Initial Phase is finished with computing time of: ')
    # with open(FinalErrStr, 'w') as of:
    #     of.write('I am going on')
    #     of.write('\n')

    for f in range(int(Freq)):
        NewCapacity += CapacityTable.iloc[f, k]
        if NewCapacity >= maxweight:
            NewCapacity = maxweight
        elif NewCapacity <= 0:
            NewCapacity = 0

        algorithm = oneplusone(solution, NewCapacity, w, p, maxp, delta, rho, cond, r,
                               int(DynamicTable[-1, NewCapacity]))
        algorithm.evaluate()  # evaluate best solution according to new capacity
        max_iterations = int(freq)
        for j in range(max_iterations):
            algorithm.mutate()
            algorithm.calc_err()
        if np.mod(f, 100) == 0:
            print('Capacity: {}, Freq: {}, FinalErr: {}'.format(NewCapacity, f, algorithm.finalerr[-1]))
        solution = copy.deepcopy(algorithm.individual)
        FinalErrList.extend(algorithm.finalerr)

        # if np.mod(f,100)==0:

    # OnePlusOne Output:
    FinalErr = np.sum(FinalErrList) / million

    with open(FinalErrStr, 'w') as of:
        of.write(str(FinalErr))
        of.write('\n')

    # print('Output is done')
