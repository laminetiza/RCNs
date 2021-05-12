import numpy as np



def r_stock_price(init, up, down, T, period_length=1, dividend_yield=0):
    """
    Computes a recombining binomial tree.
    
    For the following inputs:
        - init = 8
        - up = 2
        - down = 0.5
        - T = 3
        - dividend_yield = 0

    Returns the following matrix:
        array([[ 8., 16., 32., 64.],
               [ 0.,  4.,  8., 16.],
               [ 0.,  0.,  2.,  4.],
               [ 0.,  0.,  0.,  1.]])

    Args:
        init (float): initial price of the security at t=0
        up (float): up factor
        down (float): down factor
        T (int): maturity date in months
        period_length (float): period length in years
        dividend_yield (float): annualized dividend yield
    
    Returns:
        (np.ndarray): (T+1)x(T+1) matrix representing the recombining binomial
        tree
    """

    # create a matrix of size T+1 x T+1 modelling a tree and initialize its root
    # with init.
    N = T + 1
    st = np.zeros([N, N])
    st[0, 0] = init

    # compute the dividend factor 
    dividend = np.exp(- dividend_yield * period_length)

    for j in range(1, N):
        
        # Dividend is deduced only in times t st 0 < t < T
        if j < T:
            # Fill the first row with the up values
            st[0, j] = st[0, j-1] * up * dividend
            # Fill the remaining rows with the down values
            for i in range(1, j+1):
                st[i, j] = st[i-1, j-1] * down * dividend
        else:
            # Fill the first row with the up values
            st[0, j] = st[0, j-1] * up
            # Fill the remaining rows with the down values
            for i in range(1, j+1):
                st[i, j] = st[i-1, j-1] * down

    return st



def nr_stock_price(init, up, down, T, period_length=1, dividend_yield=0):
    """
    Computes a non-recombining binomial tree.
    
    For the following inputs:
        - init = 8
        - up = 2
        - down = 0.5
        - T = 3
        - dividend_yield = 0

    Returns the following matrix:
        array([[ 8., 16., 32., 64.],
               [ 0.,  0.,  0., 16.],
               [ 0.,  0.,  8., 16.],
               [ 0.,  0.,  0.,  4.],
               [ 0.,  4.,  8., 16.],
               [ 0.,  0.,  0.,  4.],
               [ 0.,  0.,  2.,  4.],
               [ 0.,  0.,  0.,  1.]])

    Args:
        init (float): initial price of the security at t=0
        up (float): up factor
        down (float): down factor
        T (int): maturity date in months
        period_length (float): period length in years
        dividend_yield (float): annualized dividend yield
    
    Returns:
        (np.ndarray): (2**T)x(T+1) matrix representing the non-recombining
        binomial tree
    """

    # create a matrix of size 2^T x T+1 modelling a tree and initialize its root
    # with init.
    N = T + 1
    tree_width = 2**T
    st = np.zeros([tree_width, N])
    st[0, 0] = init

    # compute the dividend factor 
    dividend = np.exp(-dividend_yield * period_length)

    # step is used to skip blank cells when populating the tree
    step = tree_width

    for j in range(1, N):
        step = step / 2
        step = int(step)

        # Dividend is deduced only in times t st 0 < t < T
        if j < T:
            for i in range(0, tree_width, 2*step):
                # UP
                st[i, j] = st[i, j-1] * up * dividend
                # DOWN
                st[i+step, j] = st[i, j-1] * down * dividend
        else:
            for i in range(0, tree_width, 2*step):
                # UP
                st[i, j] = st[i, j-1] * up
                # DOWN
                st[i+step, j] = st[i, j-1] * down
    return st



def r_price(init, up, down, T, k, r=0, period_length=1, dividend_yield=0, q=0.5,
            option_type=''):
    """
    Computes the pricing matrix and the equivalent portfolio of the risky asset
    of a European Option using a recombining binomial tree.
    
    For the following inputs:
        - init = 8
        - up = 2
        - down = 0.5
        - T = 3
        - k = 40
        - r = log(1.25)
        - period_length = 1
        - dividend_yield = 0
        - q = 0.5
        - option_type = 'put'
    
    Returns the following matrices:
        
        array([[14.016, 13.44 ,  9.6  ,  0.   ],
               [ 0.   , 21.6  , 24.   , 24.   ],
               [ 0.   ,  0.   , 30.   , 36.   ],
               [ 0.   ,  0.   ,  0.   , 39.   ]])
        
        array([[-0.68, -0.6 , -0.5 ],
               [ 0.  , -1.  , -1.  ],
               [ 0.  ,  0.  , -1.  ]])

    Args:
        init (float): initial price of the security at t=0
        up (float): up factor
        down (float): down factor
        T (int): maturity date in months
        k (float): strike price of the option
        r (float): annualized interest rate
        period_length (float): length of the period in years
        dividend_yield (float): annualized dividend yield
        q (float): risk-neutral probability
        option_type (str): type of the option, 'call' or 'put'
    
    Returns:
        (np.ndarray): (T+1)x(T+1) matrix representing the non-recombining
        binomial tree
        (np.ndarray): TxT matrix representing the equivalent portfolio of the
        risky asset
    """

    if option_type not in ('call', 'put'):
        raise UserWarning("Option type must be either 'call' or 'put' !")
    
    if q > 1 or q < 0:
        raise UserWarning("q is a probability and has to be between 0 and 1 !")

    # True if type is 'call', thus False if 'put'
    call_option = option_type == 'call'

    # -1 if put and 1 if call
    mul = 2 * call_option - 1

    # compute the stock price matrix
    stock_price_matrix = r_stock_price(init=init,
                                       up=up,
                                       down=down,
                                       T=T,
                                       period_length=period_length,
                                       dividend_yield=dividend_yield)
    
    # compute the discount factor
    discount = np.exp(- r * period_length)

    # payoff matrix of the same shape as the stock price matrix
    poff = np.zeros_like(stock_price_matrix)
    pi = np.zeros([T, T])

    # computes H at T
    poff[:,-1] = np.maximum(0, mul * (stock_price_matrix[:,-1] - k))

    # fills back the tree with the prices
    for j in reversed(range(T)):
        for i in range(j+1):
            # compute the price at time t as an expectation of future prices
            poff[i, j] = discount * (q * poff[i, j+1] + (1-q) * poff[i+1, j+1])

            # compute pi
            diff_in_payoff = poff[i, j+1] - poff[i+1, j+1]
            diff_in_price = stock_price_matrix[i, j+1] - stock_price_matrix[i+1, j+1]
            pi[i, j] = (diff_in_payoff) / (diff_in_price)

    return poff, pi
    # return poff[0,0], pi[0,0], poff[0,0]- pi[0,0] * init
    

    
def compute_barrier_tree(tree, threshold, option_type):
    """
    Computes an underlying tree signaling the paths that attained the barrier
    in previous times. (TO BE USED ONLY WITH NR-TREES !)

    For an input tree like:
        array([[ 8., 16., 32., 64.],
               [ 0.,  0.,  0., 16.],
               [ 0.,  0.,  8., 16.],
               [ 0.,  0.,  0.,  4.],
               [ 0.,  4.,  8., 16.],
               [ 0.,  0.,  0.,  4.],
               [ 0.,  0.,  2.,  4.],
               [ 0.,  0.,  0.,  1.]])
    and a threshold of 5 for a put option,

    returns the following tree:
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  8.,  0.],
               [ 0.,  0.,  0.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  0.,  0.,  1.],
               [ 0.,  0.,  1.,  1.],
               [ 0.,  0.,  0.,  1.]])

    Args:
        tree (np.ndarray): tree of prices
        threshold (float): threshold for the barrier
        option_type (str): can be either 'put' or 'call'
    
    Returns:
        (np.ndarray): of the same shape as the argument `tree` representing
        whether the barrier was attained in the path to a specific value in the
        matrix.
    """

    # get the width anf height of the tree
    width, height = tree.shape

    # create a tree of zeros of the same shape as the argument tree
    res = np.zeros_like(tree)
    
    # initialize the root of the tree depending on whether it is a put or call
    if option_type == 'put':
        res[0, 0] = tree[0, 0] < threshold
    else:
        res[0, 0] = tree[0, 0] > threshold

    # step is used to skip useless cells in the matrix (since it is sparse)
    step = width

    # iterate over all times j
    for j in range(1, height):
        step = step / 2
        step = int(step)

        # for each node at time j, check whether the threshold was attained
        for i in range(0, width, 2*step):

            # checking whether call or put
            if option_type == 'put':
                # UP
                res[i, j] = res[i, j-1] or tree[i, j] < threshold
                # DOWN
                res[i+step, j] = res[i, j-1] or tree[i+step, j] < threshold
            else:
                # UP
                res[i, j] = res[i, j-1] or tree[i, j] > threshold
                # DOWN
                res[i+step, j] = res[i, j-1] or tree[i+step, j] > threshold            
    return res



def nr_price(init, up, down, T, k, r=0, period_length=1, dividend_yield=0,
             q=0.5, option_type='', barrier_beta=-1):
    """
    Computes the pricing matrix and the equivalent portfolio of the risky asset
    of a European Option using a non-recombining binomial tree.
    
    For the following inputs:
        - init = 8
        - up = 2
        - down = 0.5
        - T = 3
        - k = 40
        - r = log(1.25)
        - period_length = 1
        - dividend_value = 0
        - q = 0.5
        - option_type = 'put'
        - barrier_beta = 1
    
    Returns the following matrices:
        
        array([[14.016, 13.44 ,  9.6  ,  0.   ],
               [ 0.   ,  0.   ,  0.   , 24.   ],
               [ 0.   ,  0.   , 24.   , 24.   ],
               [ 0.   ,  0.   ,  0.   , 36.   ],
               [ 0.   , 21.6  , 24.   , 24.   ],
               [ 0.   ,  0.   ,  0.   , 36.   ],
               [ 0.   ,  0.   , 30.   , 36.   ],
               [ 0.   ,  0.   ,  0.   , 39.   ]])
        
        array([[-0.68, -0.6 , -0.5 ],
               [ 0.  ,  0.  , -1.  ],
               [ 0.  , -1.  , -1.  ],
               [ 0.  ,  0.  , -1.  ]])

    Args:
        init (float): initial price of the security at t=0
        up (float): up factor
        down (float): down factor
        T (int): maturity date
        k (float): strike price of the option
        r (float): annualized interest rate
        period_length (float): length of the periods
        divident_yield (float): dividend yield
        q (float): risk-neutral probability
        option_type (str): type of the option, 'call' or 'put'
        barrier_beta (float): barrier for the down-and-in options
    
    Returns:
        (np.ndarray): (2**T)x(T+1) matrix representing the non-recombining
        binomial tree
        (np.ndarray): (2**(T-1))xT matrix representing the equivalent portfolio
        of the risky asset
    """

    # preliminary argument cheks
    if option_type not in ('call', 'put'):
        raise UserWarning("Option type must be either 'call' or 'put' !")

    if barrier_beta != -1 and (barrier_beta > 1 or barrier_beta < 0):
        raise UserWarning("Beta must be between 0 and 1 !")

    if q > 1 or q < 0:
        raise UserWarning("q is a probability and has to be between 0 and 1 !")

    # True if type is 'call', thus False if 'put'
    call_option = option_type == 'call'

    # -1 if put and 1 if call,
    # will be used to inverse (price - strike) when needed
    mul = 2 * call_option - 1

    # barrier threshold
    barrier_threshold = init * barrier_beta

    # compute the stock price matrix
    stock_price_matrix = nr_stock_price(init=init,
                                        up=up,
                                        down=down,
                                        T=T,
                                        period_length=period_length,
                                        dividend_yield=dividend_yield)
    
    # compute the discount factor
    discount = np.exp(- r * period_length)

    # payoff matrix of the same shape as the stock price matrix
    tree_width = stock_price_matrix.shape[0]

    poff = np.zeros_like(stock_price_matrix)
    pi = np.zeros([int(tree_width/2), T])

    # computes H at T
    if barrier_beta != -1:
        # compute tree stating whether each node's path had a value below the
        # threshold
        barrier_tree = compute_barrier_tree(stock_price_matrix,
                                            barrier_threshold,
                                            option_type)

        # compute the maximum between 0 or the payoff.
        poff[:,-1] = np.maximum(0,
                                mul * (np.multiply(stock_price_matrix[:,-1] - k,
                                                   barrier_tree[:,-1])))
    else:
        poff[:,-1] = np.maximum(0, mul * (stock_price_matrix[:,-1] - k))

    # fills back the tree with the prices
    step = 1
    for j in reversed(range(T)):
        step = step * 2
        l = [x for x in range(0, tree_width, step)]
        for i in range(0, tree_width, step):

            # compute the price at time t as an expectation of future prices
            poff[i, j] = discount * (q * poff[i, j+1] + (1-q) * poff[i+int(step/2), j+1])

            # compute pi
            pi_step = int(step / 2)
            diff_in_payoff = poff[i, j+1] - poff[i+pi_step, j+1]
            diff_in_price = stock_price_matrix[i, j+1] - stock_price_matrix[i+pi_step, j+1]
            pi[int(i/2), j] = (diff_in_payoff) / (diff_in_price)

    return poff, pi,
    # return poff[0,0], pi[0,0], poff[0,0]- pi[0,0] * init
    
    
    
def bond_value(coupon, payment_dates, r, period_length):
    """
    Compute the present value of a bond.

    Args:
        coupon (float): coupon rate
        payment_dates (list or np.ndarray): payment dates of the coupon
        r (float): annualized interest rate
        period_length (float): time between payment dates
    
    Returns:
        (np.ndarray): array of representing the discounted value of the bond at
        times from 0 to the maturity date
    """

    # maturity date
    T = payment_dates[-1]
    
    # copute discount factor and coupon per period
    discount = np.exp(- r * period_length)
    cpn = coupon * period_length
    
    # initialize the face value of the bond
    fv = 1
    
    # initialize the array of values with the face value
    # this represents value at time t=T
    price = np.array([fv * (1 + cpn)])

    # t from T-1 down to 1
    for t in reversed(range(1, T)):
        # the price at time t is price at time t+1 discounted + coupon
        price = np.append(price, price[-1] * discount + cpn * fv)
    
    # t=0, dicount price at t=1
    price = np.append(price, price[-1] * discount)

    # return reversed array so that times are from 0 to T
    return price[::-1]



def callable_price(bond_prices, short_put_prices, c, r=0, period_length=1,
                   q=0.5):
    """
    Computes price of a callable RCN, given the option pricing tree and the bond
    pricing array.

    Args:
        bond_prices (list or np.ndarray): array of bond values from t = 0 to T
        short_put_prices (np.ndarray): option pricing matrix representing the pricing tree
        c (float): annualized coupon rate
        r (float): annualized interes rate
        period_length (float): ength of the period
        q (float): risk-neutral probability
    Returns:
        (np.ndarray): matrix of the same shape as the option price matrix
        representing the pricing of the callable RCN.
    """
    # get shape of the tree
    width, height = short_put_prices.shape

    # compute discount factor and coupon per period
    discount = np.exp(- r * period_length)
    cpn = c * period_length

    # pricing tree of the same shape as the option pricing tree
    price = np.zeros_like(short_put_prices)

    # initialize the leaves with the RCN price at time T
    price[:,-1] = bond_prices[-1] - short_put_prices[:,-1]

    # It's a non-recombining tree
    step = 1
    for j in reversed(range(height-1)):
        step = step * 2
        l = [x for x in range(0, width, step)]
        for i in range(0, width, step):
            # compute payoff of node
            poff = discount * (q * price[i, j+1] + (1-q) * price[i+int(step/2), j+1])
            
            # if payoff is larger than 1, then call and the price is the pricipal which is 1 + the coupon
            if poff >= 1:
                price[i, j] = 1 + cpn
            else:
                price[i, j] = poff
    
    return price




def RCN_price(initial_index_price, up_factor, down_factor, RCN_coupon_rate,
              RCN_alpha, RCN_payment_dates, index_dividend_yield=0,
              RCN_barrier=False, RCN_callable=False, r=0, period_length=1,
              RCN_beta=-1):
    """
    Compute the price of an RCN.

    Args:
        initial_index_price (float): initial price f the underlying index
        up_factor (float): up factor for the binomial model
        down_factor (float): down factor for the binomial model
        RCN_coupon_rate (float): annualized coupon rate
        RCN_alpha (float): strike alpha
        RCN_payment_dates (list or np.ndarray): payment dates of the coupon
        index_dividend_yield (float): annualized dividend yield
        RCN_barrier (bool): whether the RCN is barrier or not (thus simple)
        RCN_callable (bool): whether the RCN is callable or not
        r (floar): annualized interest rate
        period_length (float): length of the period in months
        RCN_beta (float): beta of the barrier

    Returns:
        (float): the price of the RCN
    """
    
    # preliminary checks
    if RCN_alpha > 1 or RCN_alpha < 0:
        raise UserWarning("Beta must be between 0 and 1 !")

    if RCN_barrier and (RCN_beta > 1 or RCN_beta < 0):
        raise UserWarning("Beta must be between 0 and 1 !")

    # compute the bound value
    bond_price_array = bond_value(coupon=RCN_coupon_rate,
                                  payment_dates=RCN_payment_dates,
                                  r=r,
                                  period_length=period_length)

    # compute the option price depending on whether the RCN is barrier or not
    if RCN_barrier or RCN_callable:
        short_eu_put_matrix, pi = nr_price(init=initial_index_price,
                                           up=up_factor,
                                           down=down_factor,
                                           T=RCN_payment_dates[-1],
                                           k=RCN_alpha*initial_index_price,
                                           r=r,
                                           period_length=period_length,
                                           dividend_yield=index_dividend_yield,
                                           q=0.5,
                                           option_type='put',
                                           barrier_beta=RCN_beta) # will be ignred if -1 and result will be non barrier

        # transforms the option price to a rate / proportion
        short_eu_put_matrix = short_eu_put_matrix / initial_index_price
        
        # computes the price of the RCN depending on whether it is callable or not
        if RCN_callable:
            price = callable_price(bond_prices=bond_price_array, 
                                   short_put_prices=short_eu_put_matrix,
                                   c=RCN_coupon_rate,
                                   r=r,
                                   period_length=period_length,
                                   q=0.5)[0, 0]
        else:
            price = bond_price_array[0] - short_eu_put_matrix[0, 0]

    else:
        short_eu_put_matrix, pi = r_price(init=initial_index_price,
                                         up=up_factor,
                                         down=down_factor,
                                         T=RCN_payment_dates[-1],
                                         k=RCN_alpha*initial_index_price,
                                         r=r,
                                         period_length=period_length,
                                         dividend_yield=index_dividend_yield,
                                         q=0.5,
                                         option_type='put')
        short_eu_put_matrix = short_eu_put_matrix / initial_index_price
        price = bond_price_array[0] - short_eu_put_matrix[0, 0]
    
    # rounds up the price to 10 decimals
    return np.round(price, 10)




def get_prices_per_u(u, ot, r, Delta, delta):
    """
    Computes an array of option prices for a given value of u

    Args:
        u (float): value of u
        ot (str): option type, 'put' or 'call'
    
    Returns:
        (list): price for each value of the strike
    """
    
    prices = []
    # for each strike, compute the price of the option
    for strike in reversed(np.linspace(9000, 12000, 16)):
        price = r_price(init=11118,
                        up=u,
                        down=1/u,
                        T=12,
                        k=strike,
                        r=r,
                        period_length=Delta,
                        dividend_yield=delta,
                        q=0.5,
                        option_type=ot)[0][0, 0]
        prices.append(price)

    return prices



def grid_search(calls_real, puts_real, r, Delta, delta):
    """
    Finds the best U to fit the data

    Args:
        calls_real (np.ndarray): call prices
        puts_real (np.ndarray): put prices
    
    Returns:
        (float): optimal U
    """
    # we use values of u in the interval (1;1.2] and with a decimal precision of 4
    us = np.linspace(1, 1.2, 2001)[1:]

    # prices of put and call options
    calls = [get_prices_per_u(u, 'call', r, Delta, delta) for u in us]
    puts = [get_prices_per_u(u, 'put', r, Delta, delta) for u in us]

    # MAE of put and call prices
    calls_error = np.abs(np.subtract(calls, calls_real))
    puts_error = np.abs(np.subtract(puts, puts_real))

    # 
    error_matrix = np.append(calls_error, puts_error, axis=1)

    # compute error per u
    error_per_u = [sum(e)/len(e) for e in error_matrix]

    return us[np.argmin(error_per_u)]
