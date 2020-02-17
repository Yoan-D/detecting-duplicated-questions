# Calculate the frequency of the words

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from scipy.stats import norm, scoreatpercentile

import matplotlib as mpl
import pickle

mpl.rcParams["font.size"] = 13


def calculate_approx_mean_from_hist(bins, y):
    numbers = 0
    for idx, b in enumerate(bins):
        if b != bins[-1]:
            numbers = numbers + ((bins[idx] + bins[idx + 1]) / 2) * y[idx]
    return numbers / np.sum(y)


def iqr(a):
    # From seaborn library
    a = np.asarray(a)
    q1 = scoreatpercentile(a, 25)
    q3 = scoreatpercentile(a, 75)
    return q3 - q1


def _freedman_diaconis_bins(a):
    # From seaborn library
    # From https://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    h = 2 * iqr(a) / (len(a) ** (1 / 3))
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))


if __name__ == '__main__':

    cleaned_list_question1 = pickle.load(open("deep_cleaned_list_question1.p", "rb"))
    cleaned_list_question2 = pickle.load(open("deep_cleaned_list_question2.p", "rb"))
    counter = Counter()
    for q1, q2 in zip(cleaned_list_question1, cleaned_list_question2):
        counter.update(q1 + q2)

    most_common_dict = counter.most_common(30)
    names = [i[0] for i in most_common_dict]
    values = [i[1] for i in most_common_dict]

    plt.figure(figsize=(20, 7))
    plt.title('The 30 most common words in the text corpus')
    plt.bar(range(len(most_common_dict)), values, tick_label=names, facecolor="blue", linewidth=2)
    plt.show()

    x = list(counter.values())  # fifty_common_words = [i[0] for i in counter.most_common(50)] [v for v, k in zip(list(counter.values()), list(counter.keys())) if k not in fifty_common_words] #excluding the 50 most common words
    words = list(counter.keys())

    x = np.log10(x) + 1  # squish values
    (mu, sigma) = norm.fit(x)

    fig = plt.figure(figsize=(20, 12))

    bins = min(_freedman_diaconis_bins(x), 50)

    y, bins, patches = plt.hist(x, bins=bins, facecolor="red", density=True, alpha=0.5)

    binwidth = bins[1] - bins[0]

    z = np.linspace(bins.min(), bins.max(), 100)
    p = norm.pdf(z, mu, sigma)
    plt.plot(z, p, 'g--', linewidth=2)

    sigma_values = [0, 1, 2, 3, 4, 5, 6, 7, -1, -2]
    xticks = []

    xticks.append(mu)
    xticks.append(mu + sigma)  # first standard deviation above
    xticks.append(mu + 2 * sigma)  # second standard deviation above
    xticks.append(mu + 3 * sigma)  # third standard deviation above
    xticks.append(mu + 4 * sigma)  # fourth standard deviation above
    xticks.append(mu + 5 * sigma)  # fifth standard deviation above
    xticks.append(mu + 6 * sigma)  # fifth standard deviation above
    xticks.append(mu + 7 * sigma)  # fifth standard deviation above
    xticks.append(mu - sigma)  # first standard deviation below
    xticks.append(mu - 2 * sigma)  # second standard deviation below

    xticks_l = ['$' + str(k) + '\sigma$' if k != 0 else '$\mu$' for index, k in enumerate(sigma_values)]
    plt.xticks(xticks, xticks_l)

    plt.gca().set(title='Log Word Frequency histogram', ylabel='Density')

    density = round(sum(_y * binwidth for _y in y), 3)

    fig.text(.7, .5, '[Findings]\n'
                     ' Total area beneath density curve: {}\n'
                     ' Unique words: {}\n'
                     ' $\mu$={}\n'
                     ' $\sigma$={}\n\n'
                     ' [Calculating word occurrence percentage: y-value * bin width]\n'
                     ' Words occurring once: {}% \n'
                     ' Words occuring twice: {}% \n '
                     'Words occuring three times: {}%\n '
                     'Average number of word occurrences: {}'
             .format(density,
                     len(x),
                     round(mu, 4), round(sigma, 4),
                     round(100 * (y[0] * binwidth), 2),
                     round(100 * (y[2] * binwidth), 2),
                     round(100 * (y[4] * binwidth), 2),
                     round(np.mean(10 ** (x - 1)), 2)), size=15, ha='center', bbox=dict(facecolor='none', edgecolor='grey', linewidth=1))

    plt.show()

    print('\n Words found between 5 and 7 standard deviations above the mean:', [words[idx] for idx, _b in enumerate(x) if _b > xticks[5]])
