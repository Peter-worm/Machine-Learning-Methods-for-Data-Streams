from river import drift
from river import tree
from river import forest
from river import naive_bayes, linear_model, neighbors
import matplotlib.pyplot as plt

def prefix_sum(list):
    result = []
    current_sum = 0
    for value in list:
        current_sum += value
        result.append(current_sum)
    return result

from collections import deque

class DynamicSlidingWindow:
    def __init__(self, max_size):
        self.window = deque()
        self.max_size = max_size
        self.sum = 0  # Optional: maintain running sum for quick stats

    def set_size(self, new_size):
        self.max_size = new_size
        while len(self.window) > self.max_size:
            removed = self.window.popleft()
            self.sum -= removed

    def append(self, value):
        self.window.append(value)
        self.sum += value
        if len(self.window) > self.max_size:
            removed = self.window.popleft()
            self.sum -= removed

    def get_window(self):
        return list(self.window)

    def get_sum(self):
        return self.sum

    def get_average(self):
        return self.sum / len(self.window) if self.window else 0

    def __len__(self):
        return len(self.window)

def take_tests_for_streams_on_tree(chained_generator, all_samples_number,metric): 
    adwin = drift.ADWIN()
    kswin = drift.KSWIN()

    detectors = {'ADWIN': adwin, 'KSWIN': kswin}

    change_points = {name: [] for name in detectors}
    accuracys= {name: [] for name in detectors}

    detector_name = list(detectors.keys())[0]       
    detector= detectors[detector_name]

    i=0
    acc_sliding_window = DynamicSlidingWindow(100)

    for detector_name in detectors:
        data = chained_generator.take(all_samples_number)
        detector= detectors[detector_name]
        window_accuracys = []
        i=0
        model = tree.HoeffdingTreeClassifier()
        acc_sliding_window = DynamicSlidingWindow(100)
        for x, y in data:
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            if y_pred is not None:
                metric.update(y, y_pred)
                # overall_accuracy.update(y, y_pred)
                correct = int(y_pred == y)

                acc_sliding_window.append(correct)
                detector.update(correct)

                window_accuracys.append(acc_sliding_window.get_average())

                if detector.drift_detected:
                    change_points[detector_name].append(i)
            i+=1
            accuracys[detector_name] = window_accuracys
    return change_points,accuracys


def take_tests_for_streams_on_given_model(chained_generator, all_samples_number,metric,model_factory): 
    adwin = drift.ADWIN()
    kswin = drift.KSWIN()

    detectors = {'ADWIN': adwin, 'KSWIN': kswin}

    change_points = {name: [] for name in detectors}
    accuracys= {name: [] for name in detectors}

    detector_name = list(detectors.keys())[0]       
    detector= detectors[detector_name]

    i=0
    acc_sliding_window = DynamicSlidingWindow(100)
    for detector_name in detectors:
        data = chained_generator.take(all_samples_number)
        detector= detectors[detector_name]
        window_accuracys = []
        i=0
        model = model_factory.create_model()
        acc_sliding_window = DynamicSlidingWindow(100)
        for x, y in data:
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            if y_pred is not None:
                metric.update(y, y_pred)
                # overall_accuracy.update(y, y_pred)
                correct = int(y_pred == y)

                acc_sliding_window.append(correct)
                detector.update(correct)

                window_accuracys.append(acc_sliding_window.get_average())

                if detector.drift_detected:
                    change_points[detector_name].append(i)
            i+=1
            accuracys[detector_name] = window_accuracys
    return change_points,accuracys

def take_tests_for_streams_on_given_model_experimental(chained_generator, all_samples_number,metric,detector_factories,model_factory,detector_offset = 5,number_of_detectors = 20): 

    pool_of_detectors = {}

    for name, factory in detector_factories.items():
        pool_of_detectors[name] = [factory() for _ in range(number_of_detectors)]

    change_points = {name: [] for name in pool_of_detectors}
    accuracys= {name: [] for name in pool_of_detectors}
    raw_accuracy = []
    detector_name = list(pool_of_detectors.keys())[0]       

    i=0
    acc_sliding_window = DynamicSlidingWindow(100)
    for detector_name in pool_of_detectors:
        data = chained_generator.take(all_samples_number)
        detectors = pool_of_detectors[detector_name]
        active_detectors = []
        window_accuracys = []
        i=0
        model = model_factory.create_model()
        acc_sliding_window = DynamicSlidingWindow(100)
        for  x, y in data:
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            if y_pred is not None:
                metric.update(y, y_pred)
                # overall_accuracy.update(y, y_pred)
                correct = int(y_pred == y)
                raw_accuracy.append(correct)
                acc_sliding_window.append(correct)

                window_accuracys.append(acc_sliding_window.get_average())

                if len(active_detectors) < number_of_detectors and i% detector_offset == 0:
                    active_detectors.append(detectors[i% number_of_detectors])
                    # print(i)
                for detector in active_detectors:
                    detector.update(correct) 
                    if detector.drift_detected:
                        change_points[detector_name].append(i)
            i+=1
            accuracys[detector_name] = window_accuracys
    return change_points,accuracys,raw_accuracy

def take_tests_for_streams_on_given_model_detectors_dynamic_pool(chained_generator, all_samples_number,metric,detector_factories,model_factory,detector_offset = 5,number_of_detectors = 20): 

    pool_of_detectors = {}

    for name, factory in detector_factories.items():
        pool_of_detectors[name] = [factory()]

    change_points = {name: [] for name in pool_of_detectors}
    accuracys= {name: [] for name in pool_of_detectors}

    detector_name = list(pool_of_detectors.keys())[0]       

    i=0
    acc_sliding_window = DynamicSlidingWindow(100)
    for detector_name in pool_of_detectors:
        data = chained_generator.take(all_samples_number)
        detectors = pool_of_detectors[detector_name]
        active_detectors = []
        window_accuracys = []
        i=0
        model = model_factory.create_model()
        acc_sliding_window = DynamicSlidingWindow(100)
        for  x, y in data:
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            if y_pred is not None:
                metric.update(y, y_pred)
                # overall_accuracy.update(y, y_pred)
                correct = int(y_pred == y)

                acc_sliding_window.append(correct)

                window_accuracys.append(acc_sliding_window.get_average())
                if i%detector_offset == 0:
                    if len(active_detectors) < number_of_detectors:
                        active_detectors.append(detector_factories[detector_name]())
                    else:
                        active_detectors.pop(0)
                        active_detectors.append(detector_factories[detector_name]())
                    # print(i)
                for detector in active_detectors:
                    detector.update(correct) 
                    if detector.drift_detected:
                        change_points[detector_name].append(i)
            i+=1
            accuracys[detector_name] = window_accuracys
    return change_points,accuracys

def take_tests_for_streams_on_given_model_fast_detectors_dynamic_pool(chained_generator, all_samples_number,metric,detector_factories,model_factory,detector_offset = 5,number_of_detectors = 20): 

    pool_of_detectors = {}

    for name, factory in detector_factories.items():
        pool_of_detectors[name] = [factory()]

    change_points = {name: [] for name in pool_of_detectors}
    accuracys= {name: [] for name in pool_of_detectors}

    detector_name = list(pool_of_detectors.keys())[0]       

    i=0
    acc_sliding_window = DynamicSlidingWindow(100)

    data = chained_generator.take(all_samples_number)

    active_detectors={}
    for name in detector_factories:
        active_detectors[name] = []
    
    window_accuracys = []
    binary_accuracy = []
    i=0
    model = model_factory.create_model()
    acc_sliding_window = DynamicSlidingWindow(100)
    for  x, y in data:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        
        if y_pred is not None:
            metric.update(y, y_pred)
            # overall_accuracy.update(y, y_pred)
            correct = int(y_pred == y)
            acc_sliding_window.append(correct)
            binary_accuracy.append(correct)
            window_accuracys.append(acc_sliding_window.get_average())
            if i%detector_offset == 0:
                for name in detector_factories:
                    if len(active_detectors[name]) < number_of_detectors:
                        active_detectors[name].append(detector_factories[name]())
                    else:
                        active_detectors[name].pop(0)
                        active_detectors[name].append(detector_factories[name]())
                # print(i)
            for name in detector_factories:
                for detector in active_detectors[name]:
                    detector.update(correct) 
                    if detector.drift_detected:
                        change_points[name].append(i)
        i+=1
    for name in detector_factories:
        accuracys[name] = window_accuracys
    return change_points,accuracys,binary_accuracy


def group_close_numbers(numbers, max_gap=1):
    numbers = sorted(int(n) for n in numbers)
    intervals = []
    current_interval = [numbers[0]]

    for num in numbers[1:]:
        if num - current_interval[-1] <= max_gap:
            current_interval.append(num)
        else:
            intervals.append(current_interval)
            current_interval = [num]
    
    intervals.append(current_interval)
    return intervals

def transform_drift_detections_into_intervals(numbers,max_gap=10,min_group_size=4):
    groups = group_close_numbers(numbers, max_gap)
    intervals = []
    for group in groups:
        if len(group) > min_group_size:
            intervals.append((group[0], group[-1]))
    return intervals


def visualize_results(detected_change_points, true_drift_points,drift_widths ,accuracys):
    # print(detected_change_points)
    for name, points in detected_change_points.items():
        plt.figure(figsize=(10, 6))
        plt.plot(accuracys[name], label='Data')
        i =0
        for point in points:
            if i == 0:
                plt.axvline(x=point, color='k', linestyle='--', label='Detected Change Points')
                i=1
            else:
                plt.axvline(x=point, color='k', linestyle='--')    
            # print(point)
        i = 0    
        for i, point in enumerate(true_drift_points[:-1]):
            # if i==0:
            plt.axvline(x=point, color='r', linestyle='--', label='True Drift Point' if i == 0 else "")
            width = drift_widths[i]
            plt.axvline(x=(point+width/2), color='g', linestyle='--',label = 'Drift width' if i == 0 else "" )
            plt.axvline(x=(point-width/2), color='g', linestyle='--',)
            i=1
            # else:
            #     plt.axvline(x=point, color='r', linestyle='--', label='True Drift Point' if i == 0 else "")
            #     width = drift_widths[i]
            #     plt.axvline(x=(point+width/2), color='g', linestyle='--',label = 'Drift width'  )
            #     plt.axvline(x=(point-width/2), color='g', linestyle='--',)
        # plt.legend()
        plt.title(f'Change Detection with River for {name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Windowed Accuracy')
        plt.legend()
        plt.show()

def visualize_results_for_fast(detected_change_points, true_drift_points,drift_widths ,accuracys):
    # print(detected_change_points)
    for name, points in detected_change_points.items():
        plt.figure(figsize=(10, 6))
        plt.plot(accuracys[name], label='Data')
        
        for point in points:
            # print(point)
            plt.axvline(x=point, color='k', linestyle='--', label='Detected Change Points')
        for i, point in enumerate(true_drift_points[:-1]):
            plt.axvline(x=point, color='r', linestyle='--', label='True Drift Point' if i == 0 else "")
            width = drift_widths[i]
            plt.axvline(x=(point+width/2), color='g', linestyle='--',  )
            plt.axvline(x=(point-width/2), color='g', linestyle='--',)
        # plt.legend()
        plt.title(f'Change Detection with River for {name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.show()


def visualize_interval_differences(intervals_a, intervals_b, labels=('List A', 'List B'),name = ""):
    def plot_intervals(ax, intervals, y, color, label=None):
        for start, end in intervals:
            ax.plot([start, end], [y, y], color=color, linewidth=6, solid_capstyle='round', label=label)
            label = None  # Only show the label once in legend

    fig, ax = plt.subplots(figsize=(10, 3))

    plot_intervals(ax, intervals_a, y=1, color='blue', label=labels[0])
    plot_intervals(ax, intervals_b, y=0, color='green', label=labels[1])

    ax.set_yticks([1, 0])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time")
    ax.set_title(f"Interval Comparison {name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_drift_intervals(true_drift_points, drift_widths):
    drift_intervals = []
    for i, point in enumerate(true_drift_points[:-1]):
        width = drift_widths[i]
        start = point - width / 2
        end = point + width / 2
        drift_intervals.append((start, end))
    return drift_intervals












def take_tests_for_streams_on_given_model_experimental_real_data(stream,metric,detector_factories,model_factory,detector_offset = 5,number_of_detectors = 20): 

    pool_of_detectors = {}

    for name, factory in detector_factories.items():
        pool_of_detectors[name] = [factory() for _ in range(number_of_detectors)]

    change_points = {name: [] for name in pool_of_detectors}
    accuracys= {name: [] for name in pool_of_detectors}
    accuracy_raw = []
    detector_name = list(pool_of_detectors.keys())[0]       

    i=0
    acc_sliding_window = DynamicSlidingWindow(100)
    list_of_stream = []
    for  i, (x, y) in enumerate(stream):
        list_of_stream.append((x,y))
    for detector_name in pool_of_detectors:
        detectors = pool_of_detectors[detector_name]
        active_detectors = []
        window_accuracys = []
        i=0
        model = model_factory.create_model()
        acc_sliding_window = DynamicSlidingWindow(100)
        print(len(list_of_stream))
        for  i, (x, y) in enumerate(list_of_stream.copy()):
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            
            if y_pred is not None:
                metric.update(y, y_pred)
                # overall_accuracy.update(y, y_pred)
                correct = int(y_pred == y)
                accuracy_raw.append(correct)
                acc_sliding_window.append(correct)

                window_accuracys.append(acc_sliding_window.get_average())

                if len(active_detectors) < number_of_detectors and i% detector_offset == 0:
                    active_detectors.append(detectors[i% number_of_detectors])
                    # print(i)
                for detector in active_detectors:
                    detector.update(correct) 
                    if detector.drift_detected:
                        change_points[detector_name].append(i)
            i+=1
            accuracys[detector_name] = window_accuracys
    return change_points,accuracys,accuracy_raw

def visualize_real_stream(detected_change_points,accuracys):
    # print(detected_change_points)
    for name, points in detected_change_points.items():
        plt.figure(figsize=(10, 6)) 
        for point in points:
            # print(point)
            plt.axvline(x=point, color='k', linestyle='--', label='Detected Change Points')
    plt.plot(accuracys[name], label='Data')
    plt.title(f'Change Detection with River for {name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.show()