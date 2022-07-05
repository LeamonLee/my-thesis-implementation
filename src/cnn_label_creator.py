import numpy as np
import tqdm
import multiprocessing

class CNN_Label_Creator_Strategy1:
    def __init__(self, df, col_name, window_size=11):
        self.df = df
        self.total_rows = len(self.df)
        # self.pbar = tqdm.tqdm_notebook(total=self.total_rows)
        # self.pbar = pbar
        self.col_name = col_name
        self.window_size = window_size
    
        
    def create_labels(self, data_series):
        pass

class CNN_Label_Creator_Strategy2:
    def __init__(self, df, col_name, window_size=11):
        self.df = df
        self.total_rows = len(self.df)
        # self.pbar = tqdm.tqdm_notebook(total=self.total_rows)
        # self.pbar = pbar
        self.col_name = col_name
        self.window_size = window_size
    
    def fake_create_labels(self, data_series):
        
        print(f"Process {self.currProcess} starts: {data_series}")
        # self.pbar.update(1)
        return data_series
        
    def create_labels(self, data_series):
            """
            Data is labeled as per the logic in research paper
            Label code : BUY => 1, SELL => 0, HOLD => 2
            params :
                df => Dataframe with data
                col_name => name of column which should be used to determine strategy
            returns : numpy array with integer codes for labels with
                      size = total-(window_size)+1
            """
            
            self.currProcess = multiprocessing.current_process().name
            print("creating label with original paper strategy")
            row_counter = 0
            total_rows = len(data_series)
            labels = np.zeros(total_rows)
            print("labels[0]: ", labels[0])
            labels[:] = np.nan
            print("labels[0]: ", labels[0])
            print("len(labels): ", len(labels))
            # pbar = tqdm.tqdm_notebook(total=total_rows)
            print("Calculating labels...")
            
            window_size = self.window_size
            
            while row_counter < total_rows:
                print(f"{self.currProcess} process runs {row_counter}")
                if row_counter >= window_size - 1:
                    window_begin = row_counter - (window_size - 1)
                    window_end = row_counter
                    window_middle = int((window_begin + window_end) / 2)

                    min_ = np.inf
                    min_index = -1
                    max_ = -np.inf
                    max_index = -1
                    
                    for i in range(window_begin, window_end + 1):
                        price = data_series.iloc[i]
                        if price < min_:
                            min_ = price
                            min_index = i
                        if price > max_:
                            max_ = price
                            max_index = i
                    
                    # print("max_index: ", max_index)
                    # print("min_index: ", min_index)
                    # print("window_middle: ", window_middle)
                    
                    if max_index == window_middle:
                        labels[window_middle] = 0
                    elif min_index == window_middle:
                        labels[window_middle] = 1
                    else:
                        labels[window_middle] = 2

                row_counter = row_counter + 1
                # self.pbar.update(1)

            # self.pbar.close()
            return labels