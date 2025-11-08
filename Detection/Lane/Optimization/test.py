import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

cProfile.run('slow_function()', 'profile_output')
p = pstats.Stats('profile_output')
p.strip_dirs().sort_stats('cumtime').print_stats(10)
