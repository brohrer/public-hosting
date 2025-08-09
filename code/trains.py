def azureml_main(dataframe1=None, dataframe2=None):
    """
    Generate a year's train data.
    Returns a dataframe with three columns:
        * departure times in hours since midnight
        * arrival times in hours since midnight
        * the maximum speed between stations in km/hr
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def get_departure_interval(t):
        """
        Returns departure intervals in seconds.

        This is actually the interval between when the previous train
        arrives at Central Square and the following train departs 
        Kendall Square.

        Between 5 am and 10 pm, trains leave Kendall Square 
        in intervals of k seconds, where

        k = k_t + k_n
        k_t = 420 + 180 sin (h - 10.5) pi/4
        k_n is drawn from a Normal distribution with a mean of 0 and 
            standard deviation of k_t/10
        h is the hours since midnight.
        """
        hours = t / (60. * 60.)
        k_t = 320. + 240. * np.sin((hours - 10.5) * np.pi / 4.)
        k_n = np.random.normal(scale = k_t/10.)
        k = k_t + k_n
        return k

    def get_transit_interval(t):
        """
        Returns transit intervals in seconds.

        The time it takes each train to travel from Kendall Square
        to Central Square is c seconds, where

        c is drawn from a Normal distributionwith a mean of 90 and
        a standard deviation of 10.
        """
        c = np.random.normal(loc=90., scale=10.)
        return c

    def get_speed(c):
        """
        Returns the peak speed that the train reaches between stations
        in km/hr.

        The peak speed in km/hr is given by p, where

        p = p_n * 2.2 / (c * 3600)
        p_n is drawn from a Normal distribution with a mean of 1.7 and a
        standard deviation of .15
        """
        p_n = np.random.normal(loc=1.2, scale=.05)
        p = p_n * 1.4 * 60. * 60. / c
        return p

    # Trains start running at 5am, or 5:00.
    t_start = 5. * 60. * 60.
    # Trains stop running at 10pm, or 22:00.
    t_stop = 22. * 60. * 60.

    departures = []
    arrivals = []
    speed = []
    for _ in range(365):
        t = t_start
        while t < t_stop:
            t += get_departure_interval(t)
            departures.append(t)
            c = get_transit_interval(t)
            t += c
            arrivals.append(t)
            v = get_speed(c)
            speed.append(get_speed(c))
            
    departures = np.array(departures) / 3600.
    arrivals = np.array(arrivals) / 3600.
    speed = np.array(speed)

    data = np.concatenate((departures[:, np.newaxis], 
                           arrivals[:, np.newaxis], 
                           speed[:, np.newaxis]), axis=1)
    '''
    plt.figure(1)
    plt.plot(arrivals - departures, speed, 'k.')
    plt.figure(2)
    plt.hist(arrivals)
    plt.show()
    '''
    dataframe = pd.DataFrame(data).sort(columns=[0])
    print dataframe

    return dataframe,

if __name__ == '__main__':
    azureml_main()

