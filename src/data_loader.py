import pandas as pd 


def load_data(file_path = r"D:\ds project\Productivity Prediction\data\student_productivity_distraction_dataset_20000.csv"):
    try: 
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(data.head())

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
    return data 


if __name__ == "__main__":
    data = load_data()
    print(data.head())  