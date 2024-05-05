import re

# Define the path to your log file
log_file_path = '/Users/lwz/Downloads/pycil-log/ssre/pretrain_0samples_cifar100_50_5_1993_resnet50_rep.log'

def extract_accuracy_data(log_path):
    old_accuracy = []
    new_accuracy = []

    # Regular expression pattern to find the accuracy data lines
    pattern = re.compile(r"CNN: \{.*'old': ([\d.]+), 'new': ([\d.]+)\}")

    # Read the log file
    with open(log_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract old and new accuracy values and convert them to float
                old_acc = float(match.group(1))
                new_acc = float(match.group(2))
                old_accuracy.append(old_acc)
                new_accuracy.append(new_acc)

    return old_accuracy, new_accuracy

# Call the function with the path to the log file
old_acc, new_acc = extract_accuracy_data(log_file_path)

# Output the sequences
print("Old Accuracy Sequence:", old_acc)
print("New Accuracy Sequence:", new_acc)
