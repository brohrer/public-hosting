def main():
    """
    Visualize the outputs of the train dataset analyses.
    """

    import matplotlib.pyplot as plt
    import pandas as pd

    # The original data set
    df_0 = pd.DataFrame.from_csv('arr_dep.csv')
    # The data set with the multiplication interaction feature.
    df_times = pd.DataFrame.from_csv('arr_times_dep.csv')
    # The data set with the subtraction interaction feature.
    df_minus = pd.DataFrame.from_csv('arr_minus_dep.csv')

    alpha = .04
    # Show the relationships between the original features and 
    # the target.
    fig = plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(df_0.index, df_0.iloc[:,1], color='red', 
             marker='.', linewidth=0, alpha=alpha)
    plt.xlabel('feature 0')
    plt.ylabel('regression target')
    
    plt.subplot(1,2,2)
    plt.plot(df_0.iloc[:,0], df_0.iloc[:,1], color='blue', 
             marker='.', linewidth=0, alpha=alpha)
    plt.xlabel('feature 1')
    fig.tight_layout()
    fig.savefig('original.png')

    # Show the relationships between the multiplicative feature and 
    # the target.
    fig = plt.figure(2)
    plt.subplot(1,2,1)
    plt.plot(df_times.index, df_times.iloc[:,1], color='red', 
             marker='.', linewidth=0, alpha=alpha)
    plt.xlabel('feature 0')
    plt.ylabel('regression target')
    
    plt.subplot(1,2,2)
    plt.plot(df_times.iloc[:,2], df_times.iloc[:,1], color='orange', 
             marker='.', linewidth=0, alpha=alpha)
    plt.xlabel('feature 0 x feature 1')
    fig.tight_layout()
    fig.savefig('multiply.png')

    # Show the relationships between the subtractive feature and 
    # the target.
    fig = plt.figure(3)
    plt.subplot(1,2,1)
    plt.plot(df_minus.index, df_minus.iloc[:,1], color='red', 
             marker='.', linewidth=0, alpha=alpha)
    plt.xlabel('feature 0')
    plt.ylabel('regression target')
    
    alpha = .02
    plt.subplot(1,2,2)
    plt.plot(df_minus.iloc[:,2], df_minus.iloc[:,1], color='green', 
             marker='.', linewidth=0, alpha=alpha)
    plt.xlabel('feature 1 - feature 0')
    fig.tight_layout()
    fig.savefig('subtraction.png')
    plt.show()
    


if __name__ == '__main__':
    main()
