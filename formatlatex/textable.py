import os
import builtins
from scripts import read_json, read_text, write


class TEXTable():    
    def __init__(self, params, task, tablewidth = "column", caption = "\\missing"):
        self.params = params
        self.datasets = self.params.datasets
        self.embeddings = self.params.embeddings
        self.task = task
        self.tablewidth = tablewidth
        self.caption = caption


    def format(self):
        # paths
        template_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "table_template.tex")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")
        output_path = os.path.join(self.params.base_directory, "formatlatex", "result", self.task + ".tex")

        # table vars
        template = self.read_template(template_path)
        shorthand = read_json(shorthand_path)
        rows = getattr(self, f'construct_rows_{self.task}')() # func depends on task, refer to temporal_precicions_avg_diff
        columns = "l" + (len(rows[0][0])-1)*"c" # set latex columns
        label = "tab:" + self.task # latex label, dependent on task

        col_best_results = self.find_best_results(rows, min)

        # format content
        content = ""
        for sec in rows:
            content += "\\hline\n" #each section separated by hlines
            for row in sec:
                row = self.format_values(row, shorthand) # decimal precision and shorthands
                row = self.highlight_values(row, self.format_values(col_best_results, shorthand)) # highlight best results
                row = self.flatten_row(row) # flatten rows with sublists, join values by en-dash
                content += self.tex_row(row) # format row as row in tex table
        content += "\\hline\n"

        # replace text in template

        output = template.replace(
            "_caption", self.caption).replace(
            "_label", label).replace(
            "_columns", columns).replace(
            "_content", content)
        
        # write to file
        write(output_path, output)

###############################################    FORMAT TEXT FUNCTIONS     ################################################3

    # format list as row in tex table
    def tex_row(self, row):
        return str(" & ".join(row) + "\\\\\n")
    
    # format text as bold for tex
    def tex_bold(self, x):
        return f"\\textbf{{{x}}}"
    
    # format all items in input
    def format_values(self, input, shorthand):
        for i, item in enumerate(input):
            match type(item):
                case builtins.list:
                    input[i] = self.format_values(item, shorthand) # recurse sublists
                case builtins.int | builtins.float | builtins.complex:
                    input[i] = f"{item:.2f}" # .2 decimal precision
                case builtins.str:
                    if item in shorthand.keys():
                        input[i] = str(shorthand[item]) # format as shorthand only if string in shorthand list
        return input

    # flatten single row, join values of sublists by en-dash
    def flatten_row(self, row):
        flat_row = []

        for item in row:
            if type(item) == list:
                item = str("\u2013".join(item))
            flat_row.append(item)

        return flat_row

    # match row against test, if values and positions match transform row value by highlight, return row
    def highlight_values(self, row, test, highlight = "tex_bold"):
        result = []

        for x, y in enumerate(row):
            if type(y) == list:
                y = self.highlight_values(y, [test[x]]*len(y))
            elif y == test[x]:
                y = getattr(self, highlight)(y)
            result.append(y)

        return result

########################################################     ROW OPERATIONS     ##########################################################

    # read template, adjust if tablewidth is "text"
    def read_template(self, path):
        template = read_text(path)

        if self.tablewidth == "text":
            template = template.replace(
                "table", "table*").replace(
                "\columnwidth", "\\textwidth"
                )
            
        return template


    # get columns of list of rows
    def get_columns(self, input):
        rows = [row for sec in input for row in sec] # flatten list for sections
        columns = [[row[i] for row in rows] for i in range(len(rows[0]))] # get columns
        return columns

    # find best result for each column, append to list, return list
    def find_best_results(self, input, minmax = max):
        columns = self.get_columns(input)
        best_results = []

        for column in columns:
            best_results.append(self.find_best_result_of_column(column, minmax))

        return best_results

    # finds best result per column
    def find_best_result_of_column(self, input, minmax = max):
        # return value, stays unchanged if input does not contain numbers
        best_result = False

        # check if items in lists are lists themselves, recurses if yes
        for i, item in enumerate(input):
            if type(item) == list:
                input[i] = self.find_best_result_of_column(item, minmax)

        # filters non-number values (e.g. header of column)
        filtered_column = list(filter( lambda x: type(x) in {int, float, complex}, input))

        # finds best of numbers in column, best defined by minmax parameter
        if bool(filtered_column) == True:
            best_result = (minmax(filtered_column))

        return best_result
            
####################################################     CONSTRUCT ROWS FUNCTIONS    ################################################
    # give error if rows are not same length
    def row_length_error(self, rows):
        if (len(set([len(row) for sec in rows for row in sec]))) != 1:
            print("ERROR: Length of rows differs")

    # construct rows for average difference between best and correct prediction
    def construct_rows_temporal_precision_avg_diff(self):
        # read input
        input = {}
        for dataset in self.datasets:
            input_path = os.path.join(self.params.base_directory, "result", dataset, "split_original", "timestamp_prediction_avg.json")
            input[dataset] = read_json(input_path)

        # construct rows (rows have sections, hline between each section)
        rows = []

        firstrow = [["Methods"] + self.datasets]
        rows.append(firstrow)

        avgs = []
        for embedding in self.embeddings:
            tmp_row = []

            # embedding name
            tmp_row += [embedding]

            # load averages
            for item in [input[d][embedding] for d in self.datasets]:
                if type(item) == dict:
                    item = list(item.values())
                tmp_row += [item]

            # append line
            avgs.append(tmp_row)
        rows.append(avgs)

        return(rows)
