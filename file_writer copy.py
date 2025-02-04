# this file's purpose is to write predictions to files

# write a single line to a file
def add_line_to_file(file_path, new_line):

    with open(file_path, "a") as file:
        file.write(new_line + "\n")

def add_lines_to_file(file_path, new_lines_arr):

    for new_line in new_lines_arr:
        add_line_to_file(file_path, new_line)

def multi_model_prediction_logger(ticker, low_volume, high_volume, low_price, 
                                  high_price, low_close, high_close):
    
    arr = [ 
        "-------------------------------------------------------------------------------\n",
        f"FOR TICKER \n\n\t'{ticker}'\n",
        f"\t\tThe predicted volume range is: {low_volume} - {high_volume}",
        f"\t\tPredicted Low point for the day is: {low_price}",
        f"\t\tPredicted High point for the day is: {high_price}",
        f"Close price range for next trading day on 'SPY' is: {low_close} - {high_close}"
        "\n\n-------------------------------------------------------------------------------\n"
    ]

    add_lines_to_file("some_file.txt", arr)

multi_model_prediction_logger("SPY",123,123,123,123,123,123)
