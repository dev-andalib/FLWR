from collections import Counter
import json
import os
import numpy as np
import math
import random

def save_metrics_to_json(metrics_dict, message, client_id, output_folder="E:/New_IDS - Copy/results/"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Prepare the output dictionary with client id, message, and metrics
    partition_data = {
        "message": message,
        "metrics": metrics_dict
    }
    
    # Define the output file for this client
    output_file = os.path.join(output_folder, f"client_{client_id}_metrics.json")

    # Read existing data from the JSON file, if it exists
    all_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]  # Convert single dict to list if needed
            except json.JSONDecodeError:
                all_data = []  # Handle empty or invalid JSON file

    # Calculate the call number (number of existing entries + 1)
    call_number = len(all_data) + 1
    partition_data["call_number"] = call_number

    # Append the new data
    all_data.append(partition_data)

    # Write updated data back to the JSON file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

def print_msg(msg, output_folder="pytorchexample/printmsg", 
              output_file_prefix="msg"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the output file path
    output_file = os.path.join(output_folder, f"{output_file_prefix}.json")
    
    # Initialize an empty list if the file does not exist or if it is empty
    if not os.path.exists(output_file):
        all_data = []
    else:
        # Read the existing data from the file (if the file exists)
        with open(output_file, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]  # Ensure data is in list format
            except json.JSONDecodeError:
                all_data = []  # Handle invalid/empty JSON file
    
    # Append the new message to the data
    all_data.append({"message": msg})
    
    # Write the updated data back to the file at the end of the round
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

def get_class_distribution(partition_id, dataloader, message, 
                           output_folder="E:/New_IDS - Copy/class_dist", 
                           output_file_prefix="class_distribution"):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a Counter to store the frequency of each label
    label_counts = Counter()

    for batch in dataloader:
        # Assume batch is a tuple (features, labels) from TensorDataset
        _, labels = batch  # Unpack the tuple, ignore features
        labels = labels.cpu().numpy()  # Convert to numpy array

        # For binary classification, labels are float tensors of shape [batch, 1]
        # Convert to binary integers (0 or 1) by thresholding
        labels = (labels > 0.5).astype(int).flatten()  # Threshold at 0.5 and flatten

        # Update the Counter with the labels in the current batch
        label_counts.update(labels)

    # Convert int64 keys to Python int for JSON serialization
    label_counts_converted = {int(key): value for key, value in label_counts.items()}

    # Prepare the output dictionary for this call
    partition_data = {
        "Client no": partition_id,
        "message": message,
        "class_distribution": label_counts_converted
    }

    # Define the output file for this partition in the specified output folder
    output_file = os.path.join(output_folder, f"{output_file_prefix}_client_{partition_id}.json")

    # Read existing data from the JSON file, if it exists
    all_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = [all_data]  # Convert single dict to list if needed
            except json.JSONDecodeError:
                all_data = []  # Handle empty or invalid JSON file

    # Calculate the call number (number of existing entries + 1)
    call_number = len(all_data) + 1
    partition_data["call_number"] = call_number

    # Append the new data
    all_data.append(partition_data)

    # Write updated data back to the JSON file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)


######################### SA PART ###########################################


output_folder = "E:\SA\Metrics"


# 1. Check if the client folder exists inside "SA Metrics" and create it if not
def isFirst(client_id):
    global output_folder
    js = os.path.join(output_folder, f"{client_id}.json")
    """Check if the folder for the client exists, and create it if not."""
    # Create the base folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        return True
    
    if not os.path.exists(js):
        return True
    return False


# 2. Read the accuracy from the JSON file for the client
def read_file(client_id):
    global output_folder
    """Read the existing accuracy from the client's JSON file."""
    output_file = os.path.join(output_folder, f"{client_id}.json")
    
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
            # print_msg("It is working")
        return existing_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading {output_file}: {e}")
            return None


# 3. Save or update the accuracy in the JSON file for the client
def save_sa(client_id, accuracy, temp, update=False):
    global output_folder
    # update 
    output_file = os.path.join(output_folder, f"{client_id}.json")
    if os.path.exists(output_file):
        existing_data = read_file(client_id)
        if update:
            existing_data["accuracy"] = accuracy
            existing_data['temp'] = temp
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=4)

    #new save           
    else:
        client_data = {"accuracy": accuracy, 
                       "temp":temp}
        with open(output_file, 'w') as f:
            json.dump(client_data, f, indent=4)





# SA send model updates or not
def file_handle(client, output_dict, temp):
    if type(client) == int or type(client) == str:
        if isFirst(client): # file not created yet
             if output_dict['val_accuracy'] == None:
               acc = 0
               save_sa(client, acc, temp) # so make a copy 

             else:
              save_sa(client, output_dict['val_accuracy'], temp)   
        
        else:
            existing_data = read_file(client) # read acc
            if existing_data != None:
                prev_acc = existing_data.get('accuracy')
                
                

            curr_acc = output_dict['val_accuracy'] # get current acc
            
            if curr_acc:
                update = fl_sa(prev_acc, curr_acc, temp) # SA below this function
                save_sa(client, curr_acc, temp, update=update)
                return update     # based on sa will update or not  




def fl_sa(prev_acc, curr_acc, temp):
    if curr_acc is None or prev_acc is None:
        return False
    
    #  always accept a better solution
    if curr_acc > prev_acc:
        return True # yes accept weight from the client for aggregation
    
    else:
        # Ensure temperature is positive to avoid division by zero
        if temp <= 0:
            return False

        
        # change in accuracy and the temperature.
        exp_T = math.exp((curr_acc - prev_acc) / temp)

        # Generate a random probability between 0 and 1
        random_probability = random.random()

        # The rest of your logic now works correctly
        if exp_T > random_probability:
            return True  # yes accept weight from the client's for aggregation
        
        else:
            return False # no don't take the client's weights