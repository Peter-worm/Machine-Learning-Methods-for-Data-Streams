import random
import numpy as np
from itertools import cycle
from river.datasets import synth
from river import tree, stream, metrics, drift
from itertools import chain
from itertools import cycle
import random


def create_complex_generator(samples_per_generator, generators = None, width = 4000):
    if isinstance(width, (int, float)):
        width_list = [width] * len(generators)
    elif isinstance(width, list):
        width_list = width
    # defining number of samples
    if isinstance(samples_per_generator, list):
        if len(samples_per_generator) != len(generators):
            raise ValueError("List 'samples_per_generator' and 'generators' must be the same size.")
    else:
        # Assume 'a' is an integer; create a list of same size as 'b'
        samples_per_generator = [samples_per_generator] * len(generators)

    number_of_generators = len(generators)

    complex_generator = generators[0]
    number_of_samples_this_far = samples_per_generator[0]

    for i in range(1,number_of_generators):
        print(i)
        complex_generator = synth.ConceptDriftStream(complex_generator,
                                generators[i],
                               seed=1, position = number_of_samples_this_far, width=width_list[i-1])
        #width=(samples_per_generator[i]+samples_per_generator[i-1])/10
        print(number_of_samples_this_far)
        print(generators[i])
        number_of_samples_this_far +=samples_per_generator[i]

    return complex_generator,width_list

def random_stream_factory(random_generator,n_features,generators_active = [0]):
    generator_type_id = random.choice(generators_active)

    if generator_type_id == 0:
        return synth.Hyperplane(random_generator.randint(0,200000), n_features=n_features,n_drift_features=0,sigma=0,noise_percentage=0)
    elif generator_type_id == 1:
        return synth.RandomRBF(seed_model = random_generator.randint(0,200000), seed_sample=random_generator.randint(0,200000),
                         n_classes=2, n_features=n_features, n_centroids=20)
    elif generator_type_id == 2:
        return synth.RandomTree(seed_tree = random_generator.randint(0,200000), seed_sample=random_generator.randint(0,200000), n_classes=2,
                           n_num_features=n_features, n_cat_features=0,
                           n_categories_per_feature=2, max_tree_depth=6,
                           first_leaf_level=3, fraction_leaves_per_level=0.15)

def generate_random_chain_stream(number_of_streams = 5,number_of_features = 5,samples_per_generator = 20000, seed = None):
    if seed:
        random_seed_generator = random.Random(seed)
    else:
        random_seed_generator = random.Random()

    list_of_generators = [random_stream_factory(random_seed_generator,number_of_features) for x in range(0,number_of_streams)]
    print(list_of_generators)
    if not isinstance(samples_per_generator, list):
        temp = samples_per_generator
        samples_per_generator = [temp for x in list_of_generators]

    chained_stream,drift_widths = create_complex_generator(
        generators=list_of_generators,
        samples_per_generator=samples_per_generator
    )
    return chained_stream,drift_widths

def generate_cyclical_drift_stream(generator_pool_size = 3, number_of_generators=9,number_of_features = 5, samples_per_generator=20000,width = 4000, n_features=5, seed=None,generators_active = [0]):
    if seed:
        random_seed_generator = random.Random(seed)
    else:
        random_seed_generator = random.Random()

    list_of_generators = [random_stream_factory(random_seed_generator,number_of_features,generators_active=generators_active) for x in range(0,generator_pool_size)]
    # print(list_of_generators)
    if not isinstance(samples_per_generator, list):
        temp = samples_per_generator
        samples_per_generator = [temp for x in list_of_generators]

    cyclical_generators = []
    for num in cycle(list_of_generators):
        if len(cyclical_generators) >= number_of_generators:
            break
        cyclical_generators.append(num)

    return create_complex_generator(
        generators=cyclical_generators,
        samples_per_generator=samples_per_generator,
        width=width
    )