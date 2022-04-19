import re
import sys
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from typing import List

heart_failure = "./data/heart_failure_clinical_records_dataset.csv"
heart_attack = "./data/Heart Attack Data Set.csv"

def read_dataset(filename: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print("Error. The specified file has not been found.")
        return None
    except Exception:
        print("An unknown error occurred.")
        return None

def dataset_info(df: pd.DataFrame) -> None:
    print(f"{'-'*42}\nThe dataframe has the following structure:\n{df.head()}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Number of duplicated rows: {df.duplicated().sum()}\n{'-'*42}")
    print(f"Missing values:\n{check_missing_values(df)}\n{'-'*42}")
    print(f"Mean values:\n{df.mean()}\n{'-'*42}")
    # Creation of a single table containing the minimum and maximum values
    # on two different columns with left adjustment
    data = { 
        "" : pd.Series(df.keys().tolist()),
        "Minimum" : pd.Series(df.min().values.tolist()),
        "Maximum" : pd.Series(df.max().values.tolist())
    }
    min_max_table = pd.concat(data, axis = 1).to_string(index=False).split('\n')
    pattern = re.compile("[A-Za-z_]+")
    for line in min_max_table:
        if min_max_table.index(line) != 0:
            left_spaces = len(line) - len(line.lstrip())
            res = re.search(pattern, line).end()
            line = f"{line[:res]}{' ' * left_spaces}{line[res:]}".lstrip()
        print(line)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print(f"{'-'*42}\nCorrelation matrix:\n{df.corr('pearson')}\n{'-'*42}")


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    null_values = df.isnull().sum().sort_values(ascending=True)
    null_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=True)
    null_values = pd.concat([null_values, null_percent], axis=0)
    return null_values

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if(df.duplicated().sum() != 0):
        print(f"Duplicated rows BEFORE the drop: {df.duplicated().sum()}")
        df = df.drop_duplicates(keep="first")
        print(f"Duplicated rows AFTER the drop: {df.duplicated().sum()}")
    return df

def split_dataframe(df: pd.DataFrame, 
                    train_percentage: int = 0, 
                    test_percentage: int = 0, 
                    validate_percentage: int = 0) -> tuple[List]:
    data = df.to_numpy()
    train_quantity = int(len(data) * (train_percentage / 100))
    test_quantity = int(len(data) * (test_percentage / 100))
    validate_quantity = int(len(data) * (validate_percentage / 100))
    train = data[:train_quantity]
    test = data[train_quantity:train_quantity+test_quantity]
    validate = data[train_quantity+test_quantity:]
    print(f"Training set elements: {train_quantity}\n"
          f"Testing set elements: {test_quantity+1}\n"
          f"Validate set elements: {validate_quantity+1}")
    return train, test, validate


def display_plots(df: pd.DataFrame) -> None:
    while True:
        plt.figure(figsize=(20,10))
        print("Choose the plot to be displayed: ")
        print("1. Correlation matrix heatmap")
        print("2. Kernel Density Estimation")
        print("3. Back to main menu")
        choice = choose_option([1,2,3])
        if choice == 1:
            sb.heatmap(df.corr(),annot=True,cmap="coolwarm",linecolor='black')
            plt.show()
        elif choice == 2:
            print("Which columns you want to compare?")
            print("First column: ")
            for key in df.keys().to_list():
                print(f"{df.keys().to_list().index(key) + 1}. {key}")
            choice_1 = df.keys().to_list()[choose_option(list(range(1,len(df.keys().to_list())+1)))-1]
            print("\nSecond column: ")
            for key in df.keys().to_list():
                if key != choice_1:
                    print(f"{df.keys().to_list().index(key) + 1}. {key}")
                else:
                    print(f"{df.keys().to_list().index(key) + 1}. {key} (Chosen)")
            choices_available = list(range(1,len(df.keys().to_list())+1))
            choices_available.remove(df.keys().to_list().index(choice_1)+1)
            choice_2 = df.keys().to_list()[choose_option(choices_available)-1]
            sb.kdeplot(data=df, x=choice_1, hue=choice_2, shade="fill")
            plt.show()
        else:
            break

def choose_option(available_options: List[int]) -> int:
    option = int(input(" >> "))
    while option not in available_options:
         option = int(input("Wrong input.\n >> "))
    return option

def main() -> None:
    print(f"{'-'*10} MAIN MENU {'-'*10}")
    print("1. Read Dataset")
    print("2. Exit the program")
    if choose_option([1,2]) == 2:
        print("Goodbye!")
        sys.exit(0)
    
    print("Please choose the data you want to analyze: ")
    print("1. Heart Attack Data")
    print("2. Heart Failure Data")
    print("3. Exit the program")
    choice = choose_option([1,2,3])

    if choice == 1:
        in_file = heart_attack
    elif choice == 2:
        in_file = heart_failure
    else:
        print("Goodbye!")
        sys.exit(0)
    
    data = read_dataset(in_file)

    split_result = None
    while True:
        print(f"{'-'*10} MAIN MENU {'-'*10}")
        print("1. Dataset Info")
        print("2. Display Plots")
        print(f"3. {'Print s' if split_result else 'S'}plit Dataframe [Train|Test|Validate]")
        print("4. Exit the program")
        choice = choose_option([1,2,3,4])
        if choice == 1:
            dataset_info(data)
        elif choice == 2:
            display_plots(data)
        elif choice == 3:
            if split_result is None:
                print("Choose the percentage of the dataset to dedicate to the training set [0-100]")
                train = choose_option(list(range(0,101)))
                print(f"Choose the percentage of the dataset to dedicate to the testing set [0-{100-train}]")
                test = choose_option(list(range(0,101 - train)))
                print(f"Choose the percentage of the dataset to dedicate to the validating set [0-{100-(train+test)}]")
                validate = choose_option(list(range(0,101 - (train + test))))
                split_result = split_dataframe(data, train, test, validate)
                print("Do You want to print the results?")
                print("1. Yes")
                print("2. No")
                if choose_option([1,2]) == 1:
                    counter = 0
                    for res in split_result:
                        if counter == 0:
                            print("Training set:")
                        elif counter == 1:
                            print("Testing set:")
                        else:
                            print("Validating set:")
                        print(res)
                        counter += 1
            else:
                counter = 0
                for res in split_result:
                    if counter == 0:
                        print("Training set:")
                    elif counter == 1:
                        print("Testing set:")
                    else:
                        print("Validating set:")
                    print(res)
                    counter += 1
        else:
            print("Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
    
    
    
